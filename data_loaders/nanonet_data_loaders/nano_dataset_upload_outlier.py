from raga import *
import pandas as pd
import datetime
import ast


label_map = {
    0 : 'total',
    1 : 'merchant_name',
    2 : 'date',
    3 : 'None'
}

def replace_url(x):    
    return f'https://raga-engineering.s3.us-east-2.amazonaws.com/nanonets/20231011184534/ImageSets/{x}'

def csv_parser(csv_file):
    data_frame = pd.read_csv(csv_file)    
    data_frame["ImageId"] = data_frame["ImageId"].apply(lambda x: x)
    data_frame["ImageUri"] = data_frame["ImageId"].apply(lambda x: replace_url(x))
    return data_frame


####################################################################
## You can use csv url or download the file and use the file path ##
####################################################################

pd_data_frame = csv_parser("https://ragatesitng-dev-storage.s3.ap-south-1.amazonaws.com/datasets/nanonet/df_distance_score.csv")

########
## OR ##
########

# pd_data_frame = csv_parser("./df_distance_score.csv")


schema = RagaSchema()
schema.add("ImageId", PredictionSchemaElement())
schema.add("ImageUri", ImageUriSchemaElement())
schema.add("Detection", InferenceSchemaElement(model="nanonet_model"))
schema.add("Recognition", ObjectRecognitionSchemaElement(model="nanonet_model", label_mapping=label_map))
schema.add("distance_score", DistanceScoreSchemaElement(label_mapping=label_map, model="nanonet_model"))

run_name = f"Nano_Net_Dataset-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", run_name= run_name, profile="raga-dev-new")

cred = DatasetCreds(region="us-east-2")

dataset_name = "nano_net_dataset_27_nov_v1"

test_ds = Dataset(test_session=test_session,
                  name=dataset_name,
                  type=DATASET_TYPE.IMAGE,
                  data=pd_data_frame,
                  schema=schema,
                  creds=cred)

#load to server
test_ds.load()