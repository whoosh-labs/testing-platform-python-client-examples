from raga import *
import pandas as pd
import datetime


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

def csv_parser(db_path):

    df = pd.read_csv(db_path)
    data_frame = pd.DataFrame()
    data_frame["ImageId"] = df["id"].apply(lambda x: StringElement(x))
    data_frame["TimeOfCapture"] = df.apply(lambda row: TimeStampElement(get_timestamp_x_hours_ago(row.name)), axis=1)
    data_frame["document"] = df["article"].apply(lambda x: StringElement(x))
    data_frame["reference_summary"] = df["highlights"].apply(lambda x: StringElement(x))
    data_frame["summary"] = df["model_summary"].apply(lambda x: StringElement(x))
    data_frame["document_vector"] = df["article_embedding"]
    data_frame["reference_summary_vector"] = df["highlights_embedding"]
    data_frame["summary_vector"] = df["model_summary_embedding"]
    data_frame["user_feedback"] = df["user_feedback"].apply(lambda x: float(x))
    data_frame["rouge_score"] = df["rouge_score"].apply(lambda x: float(x))
    data_frame["meteor_score"] = df["meteor_scores"].apply(lambda x: float(x))
    data_frame["bleu_score"] = df["bleu_scores"].apply(lambda x: float(x))
    data_frame["cosine_score"] = df["cosine_scores"].apply(lambda x: float(x))
    data_frame["document_length"] = df["article"].apply(lambda x: len(x))

    return data_frame

###################################################################################################

pd_data_frame = csv_parser("/Users/rupalitripathi/IdeaProjects/testing-platform-python-client/raga/examples/assets/test_with_metrics.csv")

schema = RagaSchema()
schema.add("ImageId", PredictionSchemaElement())
schema.add("TimeOfCapture", TimeOfCaptureSchemaElement())
schema.add("document", TextSchemaElement())
schema.add("reference_summary", TextSchemaElement())
schema.add("summary", TextSchemaElement())
schema.add("document_vector", ImageEmbeddingSchemaElement(model="llm"))
schema.add("reference_summary_vector", ImageEmbeddingSchemaElement(model="gt"))
schema.add("summary_vector", ImageEmbeddingSchemaElement(model="llm"))
schema.add("user_feedback", NonIndexableNumericalFeatureSchemaElement())
schema.add("rouge_score", NonIndexableNumericalFeatureSchemaElement())
schema.add("meteor_score", NonIndexableNumericalFeatureSchemaElement())
schema.add("bleu_score", NonIndexableNumericalFeatureSchemaElement())
schema.add("cosine_score", NonIndexableNumericalFeatureSchemaElement())
schema.add("document_length", NumericalFeatureSchemaElement())

run_name = f"FMA_LLM_Dataset-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", run_name = run_name, profile="dev1")

test_ds = Dataset(test_session=test_session,
                  name="fma_llm_dataset_8Jan_dev2",
                  type=DATASET_TYPE.IMAGE,
                  data=pd_data_frame,
                  schema=schema)

# load to server
test_ds.load()
