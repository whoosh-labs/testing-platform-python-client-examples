import ast
import random
from raga import *
import pandas as pd
import datetime


label_map = {44: 'bottle',
             67: 'dining table',
             1: 'person',
             51: 'bowl',
             47: 'cup',
             62: 'chair',
             31: 'handbag',
             10: 'traffic light',
             3: 'car',
             84: 'book'}

def get_timestamp_x_hours_ago(hours):
    current_time = datetime.datetime.now()
    delta = datetime.timedelta(days=90, hours=hours)
    past_time = current_time - delta
    timestamp = int(past_time.timestamp())
    return timestamp


def annotation_v1(row):
    AnnotationsV1 = ImageDetectionObject()
    for detection in eval(row["AnnotationsV1"]):
        AnnotationsV1.add(ObjectDetection(Id=detection['Id'], ClassId=detection['labelId'], ClassName=detection['labelName'], Confidence=detection['Confidence'], BBox= detection['Segmentation'], Format="xn,yn_normalised"))
    return AnnotationsV1

def model_inferences(row):
    ModelInferences = ImageDetectionObject()
    for detection in eval(row["yolov8n-seg_model"]):
        ModelInferences.add(ObjectDetection(Id=detection['Id'], ClassId=detection['labelId'], ClassName=detection['labelName'], Confidence=detection['Confidence'], BBox= detection['Segmentation'], Format="xn,yn_normalised"))
    return ModelInferences


def csv_parser(csv_file):
    df = pd.read_csv(csv_file)
    data_frame = pd.DataFrame()
    data_frame["ImageId"] = df["ImageId"]
    data_frame["ImageUri"] = df["ImageUrl"]
    data_frame["TimeOfCapture"] = df.apply(lambda row: TimeStampElement(get_timestamp_x_hours_ago(row.name)), axis=1)
    data_frame["AnnotationsV1"] = df.apply(annotation_v1, axis=1)
    data_frame["yolov8n-seg_model"] = df.apply(model_inferences, axis=1)
    return data_frame


####################################################################
## You can use csv url or download the file and use the file path ##
####################################################################

pd_data_frame = csv_parser("/Users/rupalitripathi/IdeaProjects/testing-platform-python-client/raga/examples/assets/engg_df_with_model_final1.csv")

########
## OR ##
########

# pd_data_frame = csv_parser("./assets/combined_pred.csv")

schema = RagaSchema()
schema.add("ImageId", PredictionSchemaElement())
schema.add("ImageUri", ImageUriSchemaElement())
schema.add("TimeOfCapture", TimeOfCaptureSchemaElement())
schema.add("AnnotationsV1", InferenceSchemaElement(model="GT", label_mapping=label_map))
schema.add("yolov8n-seg_model", InferenceSchemaElement(model="yolov8"))

# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", profile="dev1")


cred = DatasetCreds(region="us-east-2")

#create test_ds object of Dataset instance
test_ds = Dataset(test_session=test_session,
                  name="fma-instance-1feb-v2",
                  type=DATASET_TYPE.IMAGE,
                  data=pd_data_frame,
                  schema=schema,
                  creds=cred)

#load schema and pandas data frame
test_ds.load()