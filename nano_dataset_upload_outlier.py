from raga import *
import pandas as pd
import datetime
import ast


label_map = {
    0 : 'total',
    1 : 'merchant_name',
    2 : 'date',
    3 : 'None'
}

def inferenceCount(x):
    inferenceCount = InferenceCount()
    x = ast.literal_eval(x)
    for key, value in x.items():
        inferenceCount.add(label_map.get(key), value)
    return inferenceCount

def dist_score(x):
    distance_score = DistanceScore()
    x = ast.literal_eval(x)
    for key, value in x.items():
        distance_score.add(key, value)
    return distance_score

def emptyInferenceCount(x):
    return InferenceCount()

def detection(row):
    Detection = ImageDetectionObject()
    row["Detection"] = ast.literal_eval(row["Detection"])
    for detection in row["Detection"]:
        Detection.add(ObjectDetection(Id=detection['Id'], ClassId=detection['ClassId'], ClassName=detection['ClassName'], Confidence=detection['Confidence'], BBox= detection['BBox'], Format="xywh_normalized"))
    return Detection

def recognition(row):
    Recognition = ImageDetectionObject()
    row["Recognition"] = ast.literal_eval(row["Recognition"])
    for recognition in row["Recognition"]:
        Recognition.add(ObjectRecognition(Id=recognition['Id'], ClassId=recognition['ClassId'], ClassName=recognition['ClassName'], Confidence=recognition['Confidence'], OcrText= recognition['OcrText']))
    return Recognition

def replace_url(s3_url):
    parts = s3_url.split('/')
    object_key = '/'.join(parts[3:])
    http_url = f'https://raga-engineering.s3.us-east-2.amazonaws.com/{object_key}'
    return http_url

def csv_parser(csv_file):
    df = pd.read_csv(csv_file)
    data_frame = pd.DataFrame()
    data_frame["ImageId"] = df["ImageId"].apply(lambda x: StringElement(x))
    data_frame["ImageUri"] = df["ImageId"].apply(lambda x: replace_url(f"s3://raga-engineering/nanonets/20231011184534/ImageSets/{x}"))
    data_frame["Detection"] = df.apply(detection, axis = 1)
    data_frame["Recognition"] = df.apply(recognition, axis = 1)
    # data_frame["actual_detections"] = df["actual_detections"].apply(inferenceCount)
    # data_frame["model_detections"] = df["actual_detections"].apply(emptyInferenceCount)
    # data_frame["distance_score"] = df["distance_score"].apply(lambda x: dist_score(x))

    return data_frame

###################################################################################################
pd_data_frame = csv_parser("./assets/dataset1_annotations(1).csv")


schema = RagaSchema()
schema.add("ImageId", PredictionSchemaElement())
schema.add("ImageUri", ImageUriSchemaElement())
schema.add("Detection", InferenceSchemaElement(model="nanonet_model"))
schema.add("Recognition", ObjectRecognitionSchemaElement(model="nanonet_model", label_mapping=label_map))
# schema.add("actual_detections", InferenceCountSchemaElement(model="nanonet_gt"))
# schema.add("model_detections", InferenceCountSchemaElement(model="nanonet_model"))
# schema.add("distance_score", DistanceScoreSchemaElement(label_mapping=label_map, model="nanonet_model"))

run_name = f"Nano_Net_Dataset-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", run_name= run_name, access_key="jncyaSPRBNfT1PeIgpoa", secret_key="2L3hE0gya2a3BgULqFGB1Y3EXQIjN29ViFlU7fjS", host="http://3.111.106.226:8080", profile="dev")

cred = DatasetCreds(region="us-east-2")
# cred = DatasetCreds(arn="arn:aws:iam::527593518644:role/raga-importer")


test_ds = Dataset(test_session=test_session,
                  name="nano_net_dataset_14_nov_v6",
                  type=DATASET_TYPE.IMAGE,
                  data=pd_data_frame,
                  schema=schema,
                  creds=cred)

#load to server
test_ds.load()


model_exe_fun = ModelExecutorFactory().get_model_executor(test_session=test_session,
                                                          model_name="nanonet_model",
                                                          version="0.1.1", wheel_path="/Users/rupalitripathi/IdeaProjects/testing-platform-python-client/dist/raga_models-0.1.8-cp39-cp39-macosx_11_0_arm64.whl")

df = model_exe_fun.execute(init_args={"device": "cpu"},
                           execution_args={"input_columns":{"imageURL":"ImageUri", "Detection":"Detection"},
                                           "output_columns":{"distance_score":"distance_score"},
                                           "column_schemas":{"distance_score":DistanceScoreSchemaElement(label_mapping=label_map, model="nanonet_model")}},
                           data_frame=test_ds)


test_ds.load()