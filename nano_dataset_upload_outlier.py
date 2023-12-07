from raga import *
import pandas as pd
import datetime
import ast

#
# label_map = {
#     0 : 'total',
#     1 : 'merchant_name',
#     2 : 'date',
#     3 : 'None'
# }
label_map = {
    0: 'Merchant_Name',
    1: 'Merchant_Address',
    2: 'Receipt_Number',
    3: 'Total_Amount',
    4: 'Tax_Amount',
    5: 'Date',
    6: 'none'
}

# label_map[6] = 'NA'

print(label_map)
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
    # http_url = f'https://raga-engineering.s3.us-east-2.amazonaws.com/{object_key}'
    http_url = f'https://prod1-testing-platform-backend-s3-storage.s3.us-east-1.amazonaws.com/{object_key}'
    print(http_url)
    return http_url

def csv_parser(csv_file):
    df = pd.read_csv(csv_file)
    data_frame = pd.DataFrame()
    # unique_datatypes = df["Recognition"].apply(type).unique()
    # print(f"Unique datatypes in Recognition column: {unique_datatypes}")
    # print("***********************")
#     column_name = 'your_column_name'  # Replace with your actual column name
#
#     # Use apply to get the types of values in each dictionary
#     value_types = df[column_name].apply(lambda x: {k: type(v) for k, v in x.items()})
#
# #    Print the result
#     print(f"Datatypes of values in '{column_name}' column:")
#     print(value_types)

    data_frame["ImageId"] = df["ImageId"].apply(lambda x: StringElement(x))
    # data_frame["ImageUri"] = df["ImageId"].apply(lambda x: replace_url(f"s3://raga-engineering/nanonets/20231011184534/ImageSets/{x}"))
    data_frame["ImageUri"] = df["ImageId"].apply(lambda x: replace_url(f"s3://raga-engineering/nanonets_3/ImageSets/{x}"))
    data_frame["Detection"] = df.apply(detection, axis = 1)
    data_frame["Recognition"] = df.apply(recognition, axis = 1)
    # data_frame["actual_detections"] = df["actual_detections"].apply(inferenceCount)
    # data_frame["model_detections"] = df["actual_detections"].apply(emptyInferenceCount)
    # data_frame["distance_score"] = df["distance_score"].apply(lambda x: dist_score(x))
# s3://raga-engineering/nanonets/979c06ba-e39b-40bf-afcc-7bef3280a4b6/ImageSets/1043-receipt-48824fff-e3a7-47b5-92f1-ce87ebfed08a.jpeg
    return data_frame

###################################################################################################
pd_data_frame = csv_parser("/Users/rupalitripathi/IdeaProjects/testing-platform-python-client/raga/examples/assets/data_small_none.csv")
# pd_data_frame =csv_parser("./assets/dataset2_annotations (2).csv")

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
# test_session = TestSession(project_name="testingProject", run_name= run_name, access_key="x7jNfqMDoP092maXj2kr", secret_key="YdljE2AN99aYgY8YHBbHS0Hed8tU1DWW4rZ2TCMr", host="http://3.111.106.226:8080", profile="dev")
test_session = TestSession(project_name="OCR", run_name= run_name, access_key="4J3dt6rGD5bq9VcvAhiL", secret_key="RPKHDiztepcrDO1kKpE7kqVm9LZ6azkjiezSzeoP", host="https://backend.prod1.ragaai.ai", profile="dev")

cred = DatasetCreds(region="us-east-1")
# cred = DatasetCreds(arn="arn:aws:iam::527593518644:role/raga-importer")

#
print(pd_data_frame.head())
# print(pd_data_frame["Recognition"][0])
# unique_datatypes = pd_data_frame["Recognition"].apply(type).unique()
# print(f"Unique datatypes in Recognition column: {unique_datatypes}")
# print("***********************")
test_ds = Dataset(test_session=test_session,
                  name="nano_net_dataset_1_dec_v27",
                  type=DATASET_TYPE.IMAGE,
                  data=pd_data_frame,
                  schema=schema,
                  creds=cred)

#load to server
test_ds.load()


model_exe_fun = ModelExecutorFactory().get_model_executor(test_session=test_session,
                                                          model_name="nanonet_model",
                                                          version="0.1.1", wheel_path="/Users/rupalitripathi/IdeaProjects/testing-platform-python-client/dist/raga_models-0.1.9-cp39-cp39-macosx_11_0_arm64.whl")

df = model_exe_fun.execute(init_args={"device": "cpu"},
                           execution_args={"input_columns":{"imageURL":"ImageUri", "Detection":"Detection"},
                                           "output_columns":{"distance_score":"distance_score"},
                                           "column_schemas":{"distance_score":DistanceScoreSchemaElement(label_mapping=label_map, model="nanonet_model")}},
                           data_frame=test_ds)

test_ds.load()