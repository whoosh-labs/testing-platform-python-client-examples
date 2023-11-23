import ast
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

def annotation_v1(row):
    AnnotationsV1 = ImageDetectionObject()
    AnnotationsV1.add(eval(row["AnnotationsV1"]))
    return AnnotationsV1

def model_inferences(row):
    ModelInferences = ImageDetectionObject()
    for detection in row["ModelInferences"]:
        ModelInferences.add(ObjectDetection(Id=detection['Id'], ClassId=detection['ClassId'], ClassName=detection['ClassName'], Confidence=detection['Confidence'], BBox= detection['BBox'], Format="xywh_normalized"))
    return ModelInferences

def imag_vectors_m1(row):
    ImageVectorsM1 = ImageEmbedding()
    row = ast.literal_eval(row["ROIVectorsM1"])
    for embedding in row:
        ImageVectorsM1.add(embedding)
    return ImageVectorsM1

def csv_parser(csv_file):
    df = pd.read_csv(csv_file).head(1)
    data_frame = pd.DataFrame()
    data_frame["ImageId"] = df["ImageId"].apply(lambda x: x)
    data_frame["ImageUri"] = df["SourceLink"].apply(lambda x: x)
    data_frame["SourceLink"] = df["SourceLink"].apply(lambda x: x)
    data_frame["TimeOfCapture"] = df.apply(lambda row: TimeStampElement(get_timestamp_x_hours_ago(row.name)), axis=1)
    data_frame["Reflection"] = df.apply(lambda row: random.choice(["Yes", "No"]), axis=1)
    data_frame["Overlap"] = df.apply(lambda row: random.choice(["Yes", "No"]), axis=1)
    data_frame["CameraAngle"] = df.apply(lambda row: random.choice(["Yes", "No"]), axis=1)
    data_frame["AnnotationsV1"] = df.apply(annotation_v1, axis=1)
    data_frame["ROIVectorsM1"] = df.apply(imag_vectors_m1, axis=1)
    return data_frame

# pd_data_frame = csv_parser("./assets/combined_gt.csv")
pd_data_frame = csv_parser("./assets/combined_pred.csv")

schema = RagaSchema()
schema.add("ImageId", PredictionSchemaElement())
schema.add("ImageUri", ImageUriSchemaElement())
schema.add("TimeOfCapture", TimeOfCaptureSchemaElement())
schema.add("SourceLink", FeatureSchemaElement())
schema.add("Reflection", AttributeSchemaElement())
schema.add("CameraAngle", AttributeSchemaElement())
schema.add("Overlap", AttributeSchemaElement())
schema.add("AnnotationsV1", InferenceSchemaElement(model="GT"))
schema.add("ROIVectorsM1", ImageEmbeddingSchemaElement(model="imageModel"))

run_name = f"run-loader-ai-dash-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", run_name= run_name, access_key="LGXJjQFD899MtVSrNHGH", secret_key="TC466Qu9PhpOjTuLu5aGkXyGbM7SSBeAzYH6HpcP", host="http://3.111.106.226:8080")


cred = DatasetCreds(region="us-east-1")

#create test_ds object of Dataset instance
test_ds = Dataset(test_session=test_session, 
                  name="drift_ai_dash_pred_object_detection_ds", 
                  type=DATASET_TYPE.IMAGE,
                  data=pd_data_frame, 
                  schema=schema,
                  creds=cred)

#load schema and pandas data frame
test_ds.load()