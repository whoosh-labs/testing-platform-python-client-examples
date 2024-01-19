from glob import glob

from raga import *
import pandas as pd
import datetime
import ast



def imag_embedding(x):
    Embeddings = ImageEmbedding()
    for embedding in x:
        Embeddings.add(Embedding(embedding))
    return Embeddings
def get_timestamp_x_hours_ago(hours):
    current_time = datetime.datetime.now()
    delta = datetime.timedelta(days=90, hours=hours)
    past_time = current_time - delta
    timestamp = int(past_time.timestamp())
    return timestamp

def replace_url(s3_url):
    parts = s3_url.split('/')
    object_key = '/'.join(parts[6:])
    http_url = f"bucket-url/{object_key}"
    return http_url


def csv_parser(db_path):
    data_w_embeddings = pd.read_parquet(db_path)
    df1 = data_w_embeddings
    data_frame = pd.DataFrame()
    data_frame["ImageId"] = df1["id"].apply(lambda x: StringElement(x))
    data_frame["TimeOfCapture"] = df1.apply(lambda row: TimeStampElement(get_timestamp_x_hours_ago(row.name)), axis=1)
    data_frame["ImageUri"] = df1["img_path"].apply(lambda x: replace_url(x))
    data_frame["embedding"] = df1["embedding"].apply(lambda x: imag_embedding(x))
    return data_frame

###################################################################################################

pd_data_frame = csv_parser("/Users/rupalitripathi/IdeaProjects/testing-platform-python-client/raga/examples/assets/train_with_embedding.parquet" )

schema = RagaSchema()
schema.add("ImageId", PredictionSchemaElement())
schema.add("TimeOfCapture", TimeOfCaptureSchemaElement())
schema.add("ImageUri", ImageUriSchemaElement())
schema.add("embedding", ImageEmbeddingSchemaElement(model="data_leakage_model"))

run_name = f"Data_Leakage_Dataset-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", run_name = run_name, profile="dev")

cred = DatasetCreds(region="us-east-2")

print(pd_data_frame.head())
test_ds = Dataset(test_session=test_session,
                  name="data_leak_dataset_200k_train_v1",
                  type=DATASET_TYPE.IMAGE,
                  data=pd_data_frame,
                  schema=schema,
                  creds=cred)

# load to server
test_ds.load()
