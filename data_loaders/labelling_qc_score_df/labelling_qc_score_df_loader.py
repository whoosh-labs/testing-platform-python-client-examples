import ast
from raga import *
import pandas as pd
import datetime

def csv_parser(csv_file):
    return pd.read_csv(csv_file)



####################################################################
## You can use csv url or download the file and use the file path ##
####################################################################

data_frame = csv_parser("https://ragatesitng-dev-storage.s3.ap-south-1.amazonaws.com/datasets/labelling_qc_score_df/labelling_qc_score_df_parsed.csv")

########
## OR ##
########

# data_frame = csv_parser("./assets/labelling_qc_score_df_parsed.csv")


schema = RagaSchema()
schema.add("ImageId", PredictionSchemaElement())
schema.add("ImageUri", ImageUriSchemaElement())
schema.add("TimeOfCapture", TimeOfCaptureSchemaElement())
schema.add("SourceLink", FeatureSchemaElement())
schema.add("GroundTruth", ImageClassificationSchemaElement(model="GT"))
schema.add("MistakeScore", MistakeScoreSchemaElement(ref_col_name="GroundTruth"))
schema.add("ImageVectorsM1", ImageEmbeddingSchemaElement(model="imageModel1"))

run_name = f"score-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", run_name= run_name, profile="dev")

cred = DatasetCreds(region="ap-south-1")

#create test_ds object of Dataset instance
test_ds = Dataset(test_session=test_session,
                  name="100-sport-dataset-loader-testing",
                  type=DATASET_TYPE.IMAGE,
                  data=data_frame, 
                  schema=schema, 
                  creds=cred)

test_ds.load()