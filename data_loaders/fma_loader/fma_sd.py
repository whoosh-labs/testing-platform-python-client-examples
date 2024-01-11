from glob import glob

from raga import *
import pandas as pd
import datetime
import ast


def detection(x, rowName):
    Detection = ImageDetectionObject()
    x = float(x)
    Detection.add(ObjectDetection(ClassName=rowName, Confidence=x))
    return Detection


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


numeric_cols =  ['loan_amnt',
                'funded_amnt',
                'funded_amnt_inv',
                'int_rate',
                'installment',
                'annual_inc',
                'dti',
                'delinq_2yrs',
                'inq_last_6mths',
                'mths_since_last_delinq',
                'open_acc',
                'pub_rec',
                'revol_bal',
                'total_acc',
                'total_pymnt',
                'total_pymnt_inv',
                'total_rec_prncp',
                'total_rec_int',
                'last_pymnt_amnt']


cat_columns = ['term', 'home_ownership', 'verification_status']


label_map = {
    "numeric_cols":numeric_cols,
    "cat_columns":cat_columns
}

def csv_parser(db_path):

    df = pd.read_csv(db_path)
    data_frame = pd.DataFrame()
    data_frame["ImageId"] = df["id"].apply(lambda x: StringElement(x))
    data_frame["TimeOfCapture"] = df.apply(lambda row: TimeStampElement(get_timestamp_x_hours_ago(row.name)), axis=1)

    data_frame["loan_amnt"] = df["loan_amnt"].apply(lambda x: float(x))
    data_frame["funded_amnt"] = df["funded_amnt"].apply(lambda x: float(x))
    data_frame["funded_amnt_inv"] = df["funded_amnt_inv"].apply(lambda x: float(x))

    data_frame["int_rate"] = df["int_rate"].apply(lambda x: float(x))
    data_frame["installment"] = df["installment"].apply(lambda x: float(x))
    data_frame["annual_inc"] = df["annual_inc"].apply(lambda x: float(x))

    data_frame["dti"] = df["dti"].apply(lambda x: float(x))
    data_frame["delinq_2yrs"] = df["delinq_2yrs"].apply(lambda x: float(x))
    data_frame["inq_last_6mths"] = df["inq_last_6mths"].apply(lambda x: float(x))

    data_frame["mths_since_last_delinq"] = df["mths_since_last_delinq"].apply(lambda x: float(x))
    data_frame["open_acc"] = df["open_acc"].apply(lambda x: float(x))
    data_frame["pub_rec"] = df["pub_rec"].apply(lambda x: float(x))

    data_frame["revol_bal"] = df["revol_bal"].apply(lambda x: float(x))
    data_frame["total_acc"] = df["total_acc"].apply(lambda x: float(x))
    data_frame["total_pymnt"] = df["total_pymnt"].apply(lambda x: float(x))

    data_frame["total_pymnt_inv"] = df["total_pymnt_inv"].apply(lambda x: float(x))
    data_frame["total_rec_prncp"] = df["total_rec_prncp"].apply(lambda x: float(x))
    data_frame["total_rec_int"] = df["total_rec_int"].apply(lambda x: float(x))

    data_frame["last_pymnt_amnt"] = df["last_pymnt_amnt"].apply(lambda x: float(x))

    data_frame["term"] = df["term"].apply(lambda x: StringElement(x))
    data_frame["home_ownership"] = df["home_ownership"].apply(lambda x: StringElement(x))
    data_frame["verification_status"] = df["verification_status"].apply(lambda x: StringElement(x))

    data_frame["repay_fail"] = df["repay_fail"].apply(lambda x: int(x))
    data_frame["model_pred"] = df["model_pred"].apply(lambda x: int(x))
    data_frame["model_pred_prob"] = df["model_pred_prob"].apply(lambda x: StringElement(x))
    return data_frame


###################################################################################################

pd_data_frame = csv_parser("/Users/rupalitripathi/IdeaProjects/testing-platform-python-client/raga/examples/assets/Anonymize_Loan_Default_data_test.csv" )

schema = RagaSchema()
schema.add("ImageId", PredictionSchemaElement())
schema.add("TimeOfCapture", TimeOfCaptureSchemaElement())
schema.add("loan_amnt", NumericalFeatureSchemaElement())
schema.add("funded_amnt", NumericalFeatureSchemaElement())
schema.add("funded_amnt_inv", NumericalFeatureSchemaElement())
schema.add("int_rate", NumericalFeatureSchemaElement())
schema.add("installment", NumericalFeatureSchemaElement())
schema.add("annual_inc", NumericalFeatureSchemaElement())
schema.add("dti", NumericalFeatureSchemaElement())
schema.add("delinq_2yrs", NumericalFeatureSchemaElement())
schema.add("inq_last_6mths", NumericalFeatureSchemaElement())
schema.add("mths_since_last_delinq", NumericalFeatureSchemaElement())
schema.add("open_acc", NumericalFeatureSchemaElement())
schema.add("pub_rec", NumericalFeatureSchemaElement())
schema.add("revol_bal", NumericalFeatureSchemaElement())
schema.add("total_acc", NumericalFeatureSchemaElement())
schema.add("total_pymnt", NumericalFeatureSchemaElement())
schema.add("total_pymnt_inv", NumericalFeatureSchemaElement())
schema.add("total_rec_prncp", NumericalFeatureSchemaElement())
schema.add("total_rec_int", NumericalFeatureSchemaElement())
schema.add("last_pymnt_amnt", NumericalFeatureSchemaElement())
schema.add("term", CategoricalFeatureSchemaElement())
schema.add("home_ownership", CategoricalFeatureSchemaElement())
schema.add("verification_status", CategoricalFeatureSchemaElement())
schema.add("repay_fail", TargetSchemaElement(label_mapping = label_map, model="gt"))
schema.add("model_pred", TargetSchemaElement(model="ModelA"))
schema.add("model_pred_prob", FeatureSchemaElement())

run_name = f"FMA_SD_Dataset-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", run_name = run_name, profile="dev1")


# print(pd_data_frame.head())
test_ds = Dataset(test_session=test_session,
                  name="fma_sd_dataset_test_embed_v25",
                  type=DATASET_TYPE.IMAGE,
                  data=pd_data_frame,
                  schema=schema)

# load to server
test_ds.load()


model_exe_fun = ModelExecutorFactory().get_model_executor(test_session=test_session,
                                                          model_name="fma-sd",
                                                          version="0.1.1", wheel_path="/Users/rupalitripathi/IdeaProjects/testing-platform-python-client/dist/raga_models-0.0.1-cp39-cp39-macosx_11_0_arm64.whl")

df = model_exe_fun.execute(init_args={"device": "cpu"},
                           execution_args={"input_columns":{"numerical_feature_column_names":numeric_cols, "categorical_feature_column_names":cat_columns},
                                           "output_columns":{"embedding":"embedding"},
                                           "column_schemas":{"embedding":ImageEmbeddingSchemaElement(model="llm_model")}},
                           data_frame=test_ds)

test_ds.raga_extracted_dataset.to_csv("nasas.csv", index = False)
test_ds.load()
