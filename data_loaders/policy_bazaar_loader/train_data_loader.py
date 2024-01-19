import ast
from raga import *
import datetime


def get_timestamp_x_hours_ago(hours):
    current_time = datetime.datetime.now()
    delta = datetime.timedelta(days=90, hours=hours)
    past_time = current_time - delta
    timestamp = int(past_time.timestamp())
    return timestamp

def ground_truth(x):
    return int(x)

def imag_embedding(x):
    x = ast.literal_eval(x)
    Embeddings = ImageEmbedding()
    for embedding in x:
        Embeddings.add(Embedding(embedding))
    return Embeddings

def csv_parser(file_path):
    df = pd.read_csv(file_path)
    data_frame = pd.DataFrame()
    data_frame["DatapointId"] = df["idx"].apply(lambda x: x)
    data_frame["TimeOfCapture"] = df.apply(lambda row: TimeStampElement(get_timestamp_x_hours_ago(row.name)), axis=1)
    data_frame["Embedding"] = df['embedding'].apply(lambda x: imag_embedding(x))
    data_frame["Target"] = df['target'].apply(lambda x: ground_truth(x))
    return data_frame



####################################################################
## You can use csv url or download the file and use the file path ##
####################################################################

data_frame = csv_parser("bucket-url/datasets/policy_bazaar/ragaAI_train_policy_bazaar.csv")

########
## OR ##
########

# data_frame = csv_parser("./ragaAI_train_policy_bazaar.csv")

schema = RagaSchema()
schema.add("DatapointId", PredictionSchemaElement())
schema.add("TimeOfCapture", TimeOfCaptureSchemaElement())
schema.add("Embedding", ImageEmbeddingSchemaElement(model="imageModel"))
schema.add("Target", TargetSchemaElement(model="GT"))

test_session = TestSession(project_name="testingProject", profile="raga-dev-new")


#create test_ds object of Dataset instance
test_ds = Dataset(test_session=test_session,
                  name="policy_bazaar_train_dataset",
                  type=DATASET_TYPE.IMAGE,
                  data=data_frame,
                  schema=schema)

# #load schema and pandas data frame
test_ds.load()