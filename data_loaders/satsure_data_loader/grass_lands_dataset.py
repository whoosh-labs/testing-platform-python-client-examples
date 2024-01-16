import pathlib

import numpy as np

from raga import *
import pandas as pd
import datetime


label_to_classname = {
    1: "no_data",
    2: "water",
    3: "trees",
    4: "grass",
    5: "flooded vegetation",
    6: "crops",
    7: "scrub",
    8: "built_area",
    9: "bare_ground",
    10: "snow_or_ice",
    11: "clouds",
}

def generate_fake_image_embedding(embedding_dimension):
    """
    Generate a single fake image embedding.

    Args:
        embedding_dimension (int): Dimensionality of the embedding.

    Returns:
        A fake image embedding as a NumPy array.
    """
    # Generate a random fake image embedding vector
    fake_embedding = np.random.rand(embedding_dimension)
    embeddings = ImageEmbedding()
    for embedding in fake_embedding:
        # print(embedding)
        embeddings.add(Embedding(embedding))
    return embeddings

def get_timestamp_x_hours_ago(hours):
    current_time = datetime.datetime.now()
    delta = datetime.timedelta(days=90, hours=hours)
    past_time = current_time - delta
    timestamp = int(past_time.timestamp())
    return timestamp

def image_url(x):
    return f"https://raga-dev-testing-platform-backend-s3-storage.s3.ap-south-1.amazonaws.com/1/GrassLands_image/{pathlib.Path(x).name}"

def mask_url(x):
    return f"https://raga-dev-testing-platform-backend-s3-storage.s3.ap-south-1.amazonaws.com/1/GrassLands/{pathlib.Path(x).name}"

def attached_label(id, loss):
    return {id:loss}

def merge_loss_dicts(loss_dicts):
    mistake_score = MistakeScore()
    for d in loss_dicts:
        for key, value in d.items():
            mistake_score.add(key=key, value=value)
    return mistake_score

def csv_parser(file_path):
    df = pd.read_csv(file_path)
    data_frame = pd.DataFrame()
    data_frame["ImageId"] = df["ImageId"].apply(lambda x: StringElement(x))
    data_frame["ImageUri"] = df["SourceLink"].apply(lambda x: StringElement(image_url(x)))
    data_frame["TimeOfCapture"] = df.apply(lambda row: TimeStampElement(get_timestamp_x_hours_ago(row.name)), axis=1)
    data_frame["SourceLink"] = df['SourceLink'].apply(lambda x:StringElement(x))
    data_frame["Annotations"] = df['Annotations'].apply(lambda x:StringElement(mask_url(x)))
    return data_frame

####################################################################
## You can use csv url or download the file and use the file path ##
####################################################################

data_frame = csv_parser("https://ragatesitng-dev-storage.s3.ap-south-1.amazonaws.com/datasets/satsure/grass_lands_dataset.csv")

########
## OR ##
########

# data_frame = csv_parser("./grass_lands_dataset.csv")


schema = RagaSchema()
schema.add("ImageId", PredictionSchemaElement())
schema.add("ImageUri", ImageUriSchemaElement())
schema.add("TimeOfCapture", TimeOfCaptureSchemaElement())
schema.add("SourceLink", FeatureSchemaElement())
schema.add("Annotations", TIFFSchemaElement(label_mapping=label_to_classname, schema="tiff"))

test_session = TestSession(project_name="testingProject", profile="raga-dev-new")
cred = DatasetCreds(region="ap-south-1")

#create test_ds object of Dataset instance
test_ds = Dataset(test_session=test_session, 
                  name="GrassLands-final", 
                  type=DATASET_TYPE.IMAGE,
                  data=data_frame, 
                  schema=schema, 
                  creds=cred)

# #load schema and pandas data frame
test_ds.load()


model_exe_fun = ModelExecutorFactory().get_model_executor(test_session=test_session, 
                                                          model_name="Satsure Embedding Model", 
                                                          version="0.1.1", wheel_path="/home/ubuntu/developments/testing-platform-python-client/raga/examples/assets/satsure/raga_models-0.1.1-cp311-cp311-linux_x86_64.whl")

df = model_exe_fun.execute(init_args={"device": "cpu"}, 
                           execution_args={"input_columns":{"img_paths":"ImageUri"}, 
                                           "output_columns":{"embedding":"ImageEmbedding"},
                                           "column_schemas":{"embedding":ImageEmbeddingSchemaElement(model="Satsure Embedding Model")}}, 
                           data_frame=test_ds)

#############################################################################################
## To store embedding generated data into csv and use it for next data loader if required. ##
## Next time you will not need to use embedding model generator
#############################################################################################
df.to_csv("grass_lands_dataset_embedding.csv", index=False)

test_ds.load()