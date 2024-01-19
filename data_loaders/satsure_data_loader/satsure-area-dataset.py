import ast
import pathlib
import random
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

def get_timestamp_x_hours_ago(hours):
    current_time = datetime.datetime.now()
    delta = datetime.timedelta(days=90, hours=hours)
    past_time = current_time - delta
    timestamp = int(past_time.timestamp())
    return timestamp

def image_url(x):
    return f"bucket-url/1/satsure_rgb/{x}"

def mask_url(x):
    return f"bucket-url/1/satsure_lulc/{x}"

def attached_label(id, loss):
    return {id:loss}

def merge_loss_dicts(x):
    mistake_score = MistakeScore()
    x = ast.literal_eval(x)
    for key, value in x.items():
        mistake_score.add(key=key, value=value.get('mistake_score'), area=value.get('area'))
    return mistake_score

def csv_parser(file_path):
    df = pd.read_csv(file_path)
    # df['loss'] = df.apply(lambda row: attached_label(row['label'], row['loss']), axis=1)
    # merged_df = df.groupby('filepath', as_index=False)['loss'].agg(merge_loss_dicts)
    data_frame = pd.DataFrame()
    data_frame["ImageId"] = df["filename"].apply(lambda x: StringElement(f"{x}.tif"))
    data_frame["ImageUri"] = df["filename"].apply(lambda x: StringElement(image_url(f"{x}.tif")))
    data_frame["TimeOfCapture"] = df.apply(lambda row: TimeStampElement(get_timestamp_x_hours_ago(row.name)), axis=1)
    data_frame["SourceLink"] = df['filename'].apply(lambda x:StringElement(f"{x}.tif"))
    data_frame["Annotations"] = df['filename'].apply(lambda x:StringElement(mask_url(f"{x}.tif")))
    data_frame["MistakeScores"] = df['mistake_score'].apply(merge_loss_dicts)
    data_frame["Reflection"] = data_frame.apply(lambda row: StringElement(random.choice(["Yes", "No"])), axis=1)
    data_frame["Overlap"] = data_frame.apply(lambda row: StringElement(random.choice(["Yes", "No"])), axis=1)
    data_frame["CameraAngle"] = data_frame.apply(lambda row: StringElement(random.choice(["Yes", "No"])), axis=1)
    return data_frame

####################################################################
## You can use csv url or download the file and use the file path ##
####################################################################

data_frame = csv_parser("https://ragatesitng-dev-storage.s3.ap-south-1.amazonaws.com/datasets/satsure/satsure-area-dataset.csv")

########
## OR ##
########

# data_frame = csv_parser("./satsure-area-dataset.csv")

schema = RagaSchema()
schema.add("ImageId", PredictionSchemaElement())
schema.add("ImageUri", ImageUriSchemaElement())
schema.add("TimeOfCapture", TimeOfCaptureSchemaElement())
schema.add("SourceLink", FeatureSchemaElement())
schema.add("Annotations", TIFFSchemaElement(label_mapping=label_to_classname, schema="tiff"))
schema.add("MistakeScores", MistakeScoreSchemaElement(ref_col_name="Annotations"))
schema.add('Reflection', AttributeSchemaElement())
schema.add('Overlap', AttributeSchemaElement())
schema.add('CameraAngle', AttributeSchemaElement())

test_session = TestSession(project_name="testingProject", profile="raga-dev-new")

cred = DatasetCreds(region="ap-south-1")

#create test_ds object of Dataset instance
test_ds = Dataset(test_session=test_session, 
                  name="satsure-area-dataset-full-v1", 
                  type=DATASET_TYPE.IMAGE,
                  data=data_frame, 
                  schema=schema)

# #load schema and pandas data frame
test_ds.load()
