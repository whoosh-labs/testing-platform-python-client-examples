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

def replace_url(old_value):
    url = f"bucket-url/1/Satellite Image dataset/{old_value.replace('./', '')}"
    return StringElement(url)

def replace_embedding(old_value):
    embeddings = ImageEmbedding()
    embeddings_list = ast.literal_eval(old_value)
    for embedding in embeddings_list:
        embeddings.add(Embedding(embedding))
    return embeddings

label_to_classname = {
    0: "Building",
	1: "Land",
	2: "Road",
	3: "Vegetation",
	4: "Water",
	5: "Unlabelled"
}

def csv_parser(csv_path):
    data_frame = pd.read_csv(csv_path)
    data_frame["ImageId"] = data_frame["ImageId"].apply(lambda x: StringElement(x))
    data_frame["ImageUri"] = data_frame['SourceLink'].apply(lambda x: replace_url(x))
    data_frame["TimeOfCapture"] = data_frame.apply(lambda row: TimeStampElement(get_timestamp_x_hours_ago(row.name)), axis=1)
    data_frame["Reflection"] = data_frame.apply(lambda row: StringElement(random.choice(["Yes", "No"])), axis=1)
    data_frame["Overlap"] = data_frame.apply(lambda row: StringElement(random.choice(["Yes", "No"])), axis=1)
    data_frame["CameraAngle"] = data_frame.apply(lambda row: StringElement(random.choice(["Yes", "No"])), axis=1)
    data_frame["SourceLink"] = data_frame['SourceLink'].apply(lambda x:StringElement(x.replace('./', '')))
    data_frame["Annotations"] = data_frame['Annotations'].apply(lambda x: replace_url(x))
    data_frame["ModelAInfernences"] = data_frame['ModelAInfernences'].apply(lambda x:replace_url(x))
    data_frame["ImageVectorsM1"] = data_frame['ImageVectorsM1'].apply(lambda x:replace_embedding(x))
    return data_frame


####################################################################
## You can use csv url or download the file and use the file path ##
####################################################################

pd_data_frame = csv_parser("https://ragatesitng-dev-storage.s3.ap-south-1.amazonaws.com/datasets/satsure/satellite_image_dataset.csv")

########
## OR ##
########

# pd_data_frame = csv_parser("./satellite_image_dataset.csv")

schema = RagaSchema()
schema.add("ImageId", PredictionSchemaElement())
schema.add("ImageUri", ImageUriSchemaElement())
schema.add("TimeOfCapture", TimeOfCaptureSchemaElement())
schema.add("Reflection", AttributeSchemaElement())
schema.add("Overlap", AttributeSchemaElement())
schema.add("CameraAngle", AttributeSchemaElement())
schema.add("SourceLink", FeatureSchemaElement())
schema.add("Annotations", TIFFSchemaElement(label_mapping=label_to_classname, schema="tiff", model="ModelA"))
schema.add("ModelAInfernences", TIFFSchemaElement(label_mapping=label_to_classname, schema="tiff", model="ModelB"))
schema.add("ImageVectorsM1", ImageEmbeddingSchemaElement(model="imageModel"))

test_session = TestSession(project_name="testingProject", profile="raga-dev-new")

cred = DatasetCreds(region="ap-south-1")

#create test_ds object of Dataset instance
test_ds = Dataset(test_session=test_session,
                  name="satellite_image_dataset",
                  type=DATASET_TYPE.IMAGE,
                  data=pd_data_frame,
                  schema=schema,
                  creds=cred)

#load to server
test_ds.load()
