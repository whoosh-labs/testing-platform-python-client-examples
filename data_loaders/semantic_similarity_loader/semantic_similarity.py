from glob import glob

from raga import *
import pandas as pd
import datetime




def replace_url(s3_url):
    parts = s3_url.split('/')
    object_key = '/'.join(parts[1:])
    http_url = f"https://raga-engineering.s3.us-east-2.amazonaws.com/satsure-sr/{object_key}"
    return http_url


def get_uri(x,df2):
    x = x.split("_")[-1]
    x = "lr_"+ x
    if not df2[df2["id"] == x].empty:
        return replace_url(df2[df2["id"] == x]["filepath"].iloc[0])
    else:
        print("no filepath in", x)

def get_embedding(x, df2):
    # Check if there are rows in df2 with the specified id
    x = x.split("_")[-1]
    x = "lr_"+ x
    if not df2[df2["id"] == x].empty:
        # If there are rows, return the embedding of the first row
        return imag_embedding(df2[df2["id"] == x]["embedding"].iloc[0])
    else:
        # If there are no rows, return a default value or handle the case accordingly
        print("no embedding in", x)

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

def get_image_id(x, df2):
    x = x.split("_")[-1]
    x = "lr_"+ x
    return StringElement(df2[df2["id"] == x]["id"].iloc[0])

def csv_parser(lr_db_path, hr_db_path):
    hr_data_w_embeddings = pd.concat(map(pd.read_parquet, glob(hr_db_path)))
    df1 = hr_data_w_embeddings
    lr_data_w_embeddings = pd.concat(map(pd.read_parquet, glob(lr_db_path)))
    df2 = lr_data_w_embeddings
    data_frame = pd.DataFrame()
    data_frame["ImageIdLr"] = df1["id"].apply(lambda x: get_image_id(x,df2))
    data_frame["ImageIdHr"] = df1["id"].apply(lambda x: StringElement(x))
    data_frame["TimeOfCapture"] = df1.apply(lambda row: TimeStampElement(get_timestamp_x_hours_ago(row.name)), axis=1)
    data_frame["ImageUriHr"] = df1["filepath"].apply(lambda x: replace_url(x))
    data_frame["ImageUriLr"] = df1["id"].apply(lambda x: get_uri(x,df2))
    return data_frame

###################################################################################################

pd_data_frame = csv_parser("/Users/rupalitripathi/IdeaProjects/testing-platform-python-client/raga/examples/assets/lr_20k/embedding_store_lr_*.gzip","/Users/rupalitripathi/IdeaProjects/testing-platform-python-client/raga/examples/assets/hr_20k/embedding_store_hr_*.gzip" )


schema = RagaSchema()
schema.add("ImageIdLr", PredictionSchemaElement())
schema.add("ImageIdHr", GeneratedImageNameSchemaElement())
schema.add("TimeOfCapture", TimeOfCaptureSchemaElement())
schema.add("ImageUriLr", ImageUriSchemaElement())
schema.add("ImageUriHr", GeneratedImageUriSchemaElement())

run_name = f"Semantic_Similarity_Dataset-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", run_name= run_name, profile="dev")

cred = DatasetCreds(region="us-east-2")
# cred = DatasetCreds(arn="arn:aws:iam::527593518644:role/raga-importer")


test_ds = Dataset(test_session=test_session,
                  name="semantic_dataset_20k_no_exe_v8",
                  type=DATASET_TYPE.IMAGE,
                  data=pd_data_frame,
                  schema=schema,
                  creds=cred)

#load to server
test_ds.load()


model_exe_fun = ModelExecutorFactory().get_model_executor(test_session=test_session,
                                                          model_name="active_learning_model",
                                                          version="0.1.1", wheel_path="/Users/rupalitripathi/IdeaProjects/testing-platform-python-client/dist/raga_models-0.1.2-cp39-cp39-macosx_11_0_arm64.whl")
#
df = model_exe_fun.execute(init_args={"device": "cpu"},
                           execution_args={"input_columns":{"img_paths":"ImageUriHr"},
                                           "output_columns":{"embedding":"hr_embedding"},
                                           "column_schemas":{"embedding":ImageEmbeddingSchemaElement(model="Satsure Embedding Model")}},
                           data_frame=test_ds)


df = model_exe_fun.execute(init_args={"device": "cpu"},
                           execution_args={"input_columns":{"img_paths":"ImageUriLr"},
                                           "output_columns":{"embedding":"lr_embedding"},
                                           "column_schemas":{"embedding":ImageEmbeddingSchemaElement(model="Satsure Embedding Model")}},
                           data_frame=test_ds)
print(df.head())



test_ds.load()
