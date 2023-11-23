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
    Annotations = eval(row["AnnotationV1"])
    AnnotationsV1.add(ObjectDetection(Id=Annotations['Id'], ClassId=Annotations['ClassId'], ClassName=Annotations['ClassName'], Confidence=Annotations['Confidence'], BBox=[], Format="xywh_normalized"))
    return AnnotationsV1


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
    for val in r:
        ImageVectorsM1.add(Embedding(val))
    return ImageVectorsM1

def roi_vectors_m1(row):
    ROIVectorsM1 = ROIEmbedding()
    r = row["ROIVectorsM1"]
    r = eval(r)
    ROIVectorsM1.add(id=1, embedding_values= r)
    return ROIVectorsM1

def generate_random():
    classes = [
        "Yes",
        "No"
    ]
    return random.choice(classes)

def csv_parser(json_file):
    df = pd.read_csv(json_file)
    data_frame = pd.DataFrame()

    data_frame["ImageId"] = df["ImageId"].apply(lambda x: StringElement(x))
    data_frame["ImageUri"] = df["SourceLink"].apply(lambda x: StringElement(x))
    data_frame["SourceLink"] = df["SourceLink"].apply(lambda x: StringElement(x))
    data_frame["TimeOfCapture"] = df.apply(lambda row: TimeStampElement(get_timestamp_x_hours_ago(row.name)), axis=1)
    data_frame["AnnotationsV1"] = df.apply(annotation_v1, axis=1)
    data_frame["ImageVectorsM1"] = df.apply(imag_vectors_m1, axis=1)
    return data_frame



####################################################################
## You can use csv url or download the file and use the file path ##
####################################################################

pd_data_frame = csv_parser("https://ragatesitng-dev-storage.s3.ap-south-1.amazonaws.com/datasets/honeywell/new_coco_data_roi.csv")

########
## OR ##
########

# pd_data_frame = csv_parser("./assets/new_coco_data_roi.csv")


schema = RagaSchema()
schema.add("ImageId", PredictionSchemaElement())
schema.add("ImageUri", ImageUriSchemaElement())
schema.add("TimeOfCapture", TimeOfCaptureSchemaElement())
schema.add("SourceLink", FeatureSchemaElement())
schema.add("AnnotationsV1", InferenceSchemaElement(model="ROIModel"))
schema.add("ImageVectorsM1", ImageEmbeddingSchemaElement(model="ROIModel"))

run_name = f"run-failure-mode-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", run_name= run_name, profile="honeywell-dev")

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