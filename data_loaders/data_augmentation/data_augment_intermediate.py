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

def replace_url(s3_url):
    parts = s3_url.split('/')
    object_key = '/'.join(parts[3:])
    # print(object_key)
    http_url = f"https://raga-engineering.s3.us-east-2.amazonaws.com/{object_key}"
    return http_url

def find_name(url):
    parts = url.split('/')
    imageName = '/'.join(parts[-1:])
    return os.path.splitext(imageName)[0]

def csv_parser(csv_file):
    df = pd.read_csv(csv_file)
    # df = df.head()
    data_frame = pd.DataFrame()
    data_frame["ImageName"] = df["SourceLink"].apply(lambda x: find_name(x))
    data_frame["SourceLink"] = df["SourceLink"].apply(lambda x: replace_url(x))
    data_frame["Output_image"] = df["Output_image"].apply(lambda x: x).apply(lambda x: replace_url(x))
    data_frame["TimeOfCapture"] = df.apply(lambda row: TimeStampElement(get_timestamp_x_hours_ago(row.name)), axis=1)
    data_frame["Weather"] = df["Weather"]
    data_frame["Time of Day"] = df["Time of Day"]
    data_frame["CustomerId"] = df["CustomerId"]

    return data_frame



####################################################################
## You can use csv url or download the file and use the file path ##
####################################################################

pd_data_frame = csv_parser("/Users/rupalitripathi/IdeaProjects/testing-platform-python-client/raga/examples/assets/Honeywell_haze_data_augmentationdata_intermediate.csv")


########
## OR ##
########

# pd_data_frame = csv_parser("./assets/combined_gt.csv")

schema = RagaSchema()
schema.add("ImageName", PredictionSchemaElement())
schema.add("SourceLink", ImageUriSchemaElement())
schema.add("TimeOfCapture", TimeOfCaptureSchemaElement())
schema.add("Output_image", GeneratedImageUriSchemaElement())
schema.add("Weather", AttributeSchemaElement())
schema.add("Time of Day", AttributeSchemaElement())
schema.add("CustomerId", AttributeSchemaElement())


# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", profile="dev1")


cred = DatasetCreds(region="us-east-2")

#create test_ds object of Dataset instance
test_ds = Dataset(test_session=test_session,
                  name="data_augment_9Feb_haze_v2",
                  type=DATASET_TYPE.IMAGE,
                  data=pd_data_frame,
                  schema=schema,
                  creds=cred)

#load schema and pandas data frame
test_ds.load()