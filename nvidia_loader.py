import ast
import random
from raga import *
import pandas as pd
import datetime

import pdb


def get_timestamp_x_hours_ago(hours):
    current_time = datetime.datetime.now()
    delta = datetime.timedelta(days=90, hours=hours)
    past_time = current_time - delta
    timestamp = int(past_time.timestamp())
    return timestamp

# [{'Id': '001221', 'ClassId': 0, 'ClassName': 'forklift', 'BBox': [0.37629296874999996, 0.6172027777777778, 0.12179999999999999, 0.18810555555555553], 'Format': 'xywh_normalized', 'Confidence': 0.8615050743774972}]
def annotation_v1(row):
    AnnotationsV1 = ImageDetectionObject()
    row = ast.literal_eval(row["AnnotationsV1"])
    for detection in row:
        AnnotationsV1.add(
            ObjectDetection(
                Id=detection["Id"],
                ClassId=detection["ClassId"],
                ClassName=detection["ClassName"],
                Confidence=detection["Confidence"],
                BBox=detection["BBox"],
                Format="xywh_normalized",
            )
        )
    return AnnotationsV1


def model_inferences(row):
    ModelInferences = ImageDetectionObject()
    row = ast.literal_eval(row["ModelInferences"])
    for detection in row:
        ModelInferences.add(
            ObjectDetection(
                Id=detection["Id"],
                ClassId=detection["ClassId"],
                ClassName=detection["ClassName"],
                Confidence=detection["Confidence"],
                BBox=detection["BBox"],
                Format="xywh_normalized",
            )
        )
    return ModelInferences


def imag_vectors_m1(row):
    ImageVectorsM1 = ImageEmbedding()
    row = ast.literal_eval(row["ImageVectorsM1"])
    for embedding in row:
        ImageVectorsM1.add(embedding)
    return ImageVectorsM1


def csv_parser(csv_file):
    df = pd.read_csv(csv_file)

    # pdb.set_trace()

    data_frame = pd.DataFrame()
    data_frame["ImageId"] = df["ImageId"].apply(lambda x: x)
    data_frame["ImageUri"] = df["SourceLink"].apply(lambda x: x)
    data_frame["SourceLink"] = df["SourceLink"].apply(lambda x: x)
    data_frame["TimeOfCapture"] = df.apply(
        lambda row: TimeStampElement(get_timestamp_x_hours_ago(row.name)), axis=1
    )
    data_frame["Reflection"] = df.apply(
        lambda row: random.choice(["Yes", "No"]), axis=1
    )
    data_frame["Overlap"] = df.apply(lambda row: random.choice(["Yes", "No"]), axis=1)
    data_frame["CameraAngle"] = df.apply(
        lambda row: random.choice(["Yes", "No"]), axis=1
    )
    data_frame["AnnotationsV1"] = df.apply(annotation_v1, axis=1)
    data_frame["ModelInferences"] = df.apply(model_inferences, axis=1)
    data_frame["ImageEmbeddings"] = df.apply(imag_vectors_m1, axis=1)
    return data_frame


####################################################################
## You can use csv url or download the file and use the file path ##
####################################################################

pd_data_frame = csv_parser("./assets/nvidia_v2.csv")


########
## OR ##
########

# pd_data_frame = csv_parser("./assets/combined_gt.csv")

schema = RagaSchema()
schema.add("ImageId", PredictionSchemaElement())
schema.add("ImageUri", ImageUriSchemaElement())
schema.add("TimeOfCapture", TimeOfCaptureSchemaElement())
schema.add("SourceLink", FeatureSchemaElement())
schema.add("AnnotationsV1", InferenceSchemaElement(model="GT"))
schema.add("ModelInferences", InferenceSchemaElement(model="ModelA"))
schema.add("ImageEmbeddings", ImageEmbeddingSchemaElement(model="imageModel"))

run_name = f"run-loader-nvidia-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

# create test_session object of TestSession instance
test_session = TestSession(
    project_name="testingProject", run_name=run_name, profile="dev"
)


cred = DatasetCreds(region="us-east-2")

# create test_ds object of Dataset instance
test_ds = Dataset(
    test_session=test_session,
    name="nvidia_loader-v1",
    type=DATASET_TYPE.IMAGE,
    data=pd_data_frame,
    schema=schema,
    creds=cred,
)

# load schema and pandas data frame
test_ds.load()
