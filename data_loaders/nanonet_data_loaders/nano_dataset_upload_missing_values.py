
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
    http_url = f'bucket-url/{object_key}'
    return http_url

def csv_parser(csv_file):
    df = pd.read_csv(csv_file)
    data_frame = pd.DataFrame()
    data_frame["ImageId"] = df["ImageId"].apply(lambda x: StringElement(x))
    data_frame["ImageUri"] = df["ImageId"].apply(lambda x: replace_url(f"s3://raga-engineering/nanonets/20231011184534/ImageSets/{x}"))
    data_frame["Detection"] = df.apply(detection, axis = 1)
    data_frame["Recognition"] = df.apply(recognition, axis = 1)

    return data_frame


####################################################################
## You can use csv url or download the file and use the file path ##
####################################################################

pd_data_frame = csv_parser("https://ragatesitng-dev-storage.s3.ap-south-1.amazonaws.com/datasets/nanonet/dataset1_annotations.csv")

########
## OR ##
########

# pd_data_frame = csv_parser("./dataset1_annotations.csv")


schema = RagaSchema()
schema.add("ImageId", PredictionSchemaElement())
schema.add("ImageUri", ImageUriSchemaElement())
schema.add("Detection", InferenceSchemaElement(model="nanonet_model"))
schema.add("Recognition", ObjectRecognitionSchemaElement(model="nanonet_model", label_mapping=label_map))

# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", profile="raga-dev-new")

cred = DatasetCreds(region="us-east-2")

test_ds = Dataset(test_session=test_session,
                  name="nano_net_dataset_14_nov_v1",
                  type=DATASET_TYPE.IMAGE,
                  data=pd_data_frame,
                  schema=schema,
                  creds=cred)

#load to server
test_ds.load()