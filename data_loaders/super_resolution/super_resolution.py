from raga import *
import pandas as pd
import datetime

def csv_parser(csv_path):
    data_frame = pd.read_csv(csv_path)
    return data_frame


####################################################################
## You can use csv url or download the file and use the file path ##
####################################################################

data_frame = csv_parser("https://ragatesitng-dev-storage.s3.ap-south-1.amazonaws.com/datasets/super_resolution/super_resolution_data_test_with_embeddings.csv")

########
## OR ##
########

# data_frame = csv_parser("./assets/super_resolution_data_test_with_embeddings.csv")


schema = RagaSchema()
schema.add("ImageId", PredictionSchemaElement())
schema.add("ImageUri", ImageUriSchemaElement())
schema.add("TimeOfCapture", TimeOfCaptureSchemaElement())
schema.add("SourceLink", FeatureSchemaElement())
schema.add("SuperImageUri", BlobSchemaElement())
schema.add("imageEmbedding", ImageEmbeddingSchemaElement(model="Satsure Embedding Model"))





run_name = f"loader_failure_mode_analysis_semantic_segmentation-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", run_name= run_name, profile="dev2")
cred = DatasetCreds(region="ap-south-1")

#create test_ds object of Dataset instance
test_ds = Dataset(test_session=test_session,
                  name="super_resolution_data_v3",
                  type=DATASET_TYPE.IMAGE,
                  data=data_frame,
                  schema=schema,
                  creds=cred)

#load to server
test_ds.load()