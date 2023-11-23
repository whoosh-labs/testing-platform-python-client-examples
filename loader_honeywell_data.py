import random
from raga import *
import pandas as pd
import datetime

def get_timestamp_x_hours_ago(hours):
    current_time = datetime.datetime.now()
    delta = datetime.timedelta(days=90, hours=hours)
    past_time = current_time - delta
    timestamp = int(past_time.timestamp())
    return timestamp

def img_url(x):
    file = x.split("/")[-1]
    return StringElement(f"https://raga-product-testing-vijay.s3.amazonaws.com/Honeywell/video_object_detection_dataset/Traffic_Videos/potted_plant/{file}")

def annotation_v1(row):
    AnnotationsV1 = ImageDetectionObject()
    # print(row[0])
    Annotations = eval(row["AnnotationV1"])
    # temp = (ObjectDetection)(row["AnnotationsV1"])
    # print(temp)
    AnnotationsV1.add(ObjectDetection(Id=Annotations['Id'], ClassId=Annotations['ClassId'], ClassName=Annotations['ClassName'], Confidence=Annotations['Confidence'], BBox=[], Format="xywh_normalized"))
    return AnnotationsV1
    # AnnotationsV1 = ImageDetectionObject()
    # Annotations = eval(row["AnnotationV1"])
    #
    # # Create the ObjectDetection object and add it to AnnotationsV1
    # object_detection = ObjectDetection(
    #     # Id=Annotations.Id,
    #     ClassId=Annotations.ClassId,
    #     ClassName=Annotations.ClassName,
    #     Confidence=Annotations.Confidence,
    #     BBox=Annotations.BBox,
    #     Format=Annotations.Format
    # )
    #
    # AnnotationsV1.add(object_detection)
    # return AnnotationsV1

def model_inferences(row):
    ModelInferences = ImageDetectionObject()
    for detection in row["ModelInferences"]:
        ModelInferences.add(ObjectDetection(Id=detection['Id'], ClassId=detection['ClassId'], ClassName=detection['ClassName'], Confidence=detection['Confidence'], BBox= detection['BBox'], Format="xywh_normalized"))
    return ModelInferences

def imag_vectors_m1(row):
    ImageVectorsM1 = ImageEmbedding()
    r = row["ROIVectorsM1"]
    r = eval(r)
    r = r[1:-1]
    # r = [float(i.strip()) for i in r.split(",")]
    # print(r)
    for val in r:
        ImageVectorsM1.add(Embedding(val))
    # for embedding in row["ROIVectorsM1"]:
    #     ImageVectorsM1.add(Embedding(embedding))
    return ImageVectorsM1
    # AnnotationsV1 = ImageDetectionObject()
    # Annotations = eval(row["AnnotationsV1"])
    #
    # # Create the ObjectDetection object and add it to AnnotationsV1
    # object_detection = ObjectDetection(
    #     Id=Annotations.Id,
    #     ClassId=Annotations.ClassId,
    #     ClassName=Annotations.ClassName,
    #     Confidence=Annotations.Confidence,
    #     BBox=Annotations.BBox,
    #     Format=Annotations.Format
    # )
    #
    # AnnotationsV1.add(object_detection)
    # return AnnotationsV1

def roi_vectors_m1(row):
    ROIVectorsM1 = ROIEmbedding()
    r = row["ROIVectorsM1"]
    r = eval(r)
    # r = [float(i.strip()) for i in r.split(",")]
    # print(r)
    ROIVectorsM1.add(id=1, embedding_values= r)
    return ROIVectorsM1

def generate_random():
    classes = [
        "Yes",
        "No"
    ]
    return random.choice(classes)

def json_parser(json_file):
    df = pd.read_csv(json_file)
    data_frame = pd.DataFrame()

    data_frame["ImageId"] = df["ImageId"].apply(lambda x: StringElement(x))
    data_frame["ImageUri"] = df["SourceLink"].apply(lambda x: StringElement(x))
    data_frame["SourceLink"] = df["SourceLink"].apply(lambda x: StringElement(x))
    data_frame["TimeOfCapture"] = df.apply(lambda row: TimeStampElement(get_timestamp_x_hours_ago(row.name)), axis=1)
    data_frame["AnnotationsV1"] = df.apply(annotation_v1, axis=1)
    data_frame["ImageVectorsM1"] = df.apply(imag_vectors_m1, axis=1)
    return data_frame

# pd_data_frame = json_parser("./assets/new_coco_data_roi.csv")
# pd_data_frame = json_parser("./assets/new_vaccination.csv")
pd_data_frame = json_parser("./assets/new_traffic.csv")
print(pd_data_frame.shape)


#### Want to see in csv file uncomment line below ####
# data_frame_extractor(json_parser("./assets/COCO_engg_final_updated.json")).to_csv("./assets/COCO_engg_final_updated_100.csv", index=False)

schema = RagaSchema()
schema.add("ImageId", PredictionSchemaElement())
schema.add("ImageUri", ImageUriSchemaElement())
schema.add("TimeOfCapture", TimeOfCaptureSchemaElement())
schema.add("SourceLink", FeatureSchemaElement())
schema.add("AnnotationsV1", InferenceSchemaElement(model="ROIModel"))
schema.add("ImageVectorsM1", ImageEmbeddingSchemaElement(model="ROIModel"))

run_name = f"run-failure-mode-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", run_name= run_name, access_key="LGXJjQFD899MtVSrNHGH", secret_key="TC466Qu9PhpOjTuLu5aGkXyGbM7SSBeAzYH6HpcP", host="http://3.111.106.226:8080")

# test_session = TestSession(project_name="testingProject", run_name= run_name, access_key="GBEAALGlGsq7HrtU8M2c", secret_key="kpFKGZcP7Q0e1ONEH0kZNOIMS19G4P1f6710ddC0", host="http://13.126.220.245:8080")

cred = DatasetCreds(region="us-east-1")

# create test_ds object of Dataset instance
test_ds = Dataset(test_session=test_session,
                  name="drift_honeywell_all_gt_object_detection",
                  type=DATASET_TYPE.IMAGE,
                  data=pd_data_frame,
                  schema=schema,
                  creds=cred)

#load schema and pandas data frame
test_ds.load()