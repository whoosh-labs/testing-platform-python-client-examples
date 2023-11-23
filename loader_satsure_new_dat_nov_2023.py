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
    http_url = f'https://raga-engineering.s3.us-east-2.amazonaws.com/{object_key}'
    return http_url

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

def csv_parser(csv_path):
    data_frame = pd.read_csv(csv_path)
    data_frame["ImageId"] = data_frame["id"].apply(lambda x: x)
    data_frame["ImageUri"] = data_frame['image_path'].apply(lambda x: replace_url(x))
    data_frame["TimeOfCapture"] = data_frame.apply(lambda row: TimeStampElement(get_timestamp_x_hours_ago(row.name)), axis=1)
    data_frame["Annotations"] = data_frame['label_path'].apply(lambda x: replace_url(x))
    return data_frame


pd_data_frame = csv_parser("./assets/satsure_new_dat_nov_2023.csv")


schema = RagaSchema()
schema.add("ImageId", PredictionSchemaElement())
schema.add("ImageUri", ImageUriSchemaElement())
schema.add("TimeOfCapture", TimeOfCaptureSchemaElement())
schema.add("Annotations", TIFFSchemaElement(label_mapping=label_to_classname, schema="tiff", model="ModelA"))


run_name = f"loader_satsure_new_dat_nov_2023-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

# satsure-prod
# test_session = TestSession(project_name="Satellite Imagery", run_name = run_name, profile="satsure-prod")
test_session = TestSession(project_name="testingProject", run_name = run_name, profile="dev")


cred = DatasetCreds(region="us-east-2")

#create test_ds object of Dataset instance
test_ds = Dataset(test_session=test_session,
                  name="satsure_new_data_nov_9_v1",
                  type=DATASET_TYPE.IMAGE,
                  data=pd_data_frame,
                  schema=schema,
                  creds=cred)

#load to server
test_ds.load()


model_exe_fun = ModelExecutorFactory().get_model_executor(test_session=test_session, 
                                                          model_name="Satsure Mistake Score Model", 
                                                          version="0.1.1", wheel_path="/home/ubuntu/developments/Annotation-Consistency-Package/dist/raga_models-0.1.7.32-py3-none-any.whl")

df = model_exe_fun.execute(init_args={"device": "cpu", 
                                      "image_folders":["/home/ubuntu/developments/datasets/new_dat_nov_2023/44_p_256_new"], 
                                      "annotation_folders":["/home/ubuntu/developments/datasets/new_dat_nov_2023/lulc_main_l2a_new"]}, 
                           execution_args={"input_columns":{"img_paths":"ImageUri"}, 
                                           "output_columns":{"mistake_score":"MistakeScores"},
                                           "column_schemas":{"mistake_score":MistakeScoreSchemaElement(ref_col_name="Annotations")}}, 
                           data_frame=test_ds)
df.to_csv("test_2.csv", index=False)
test_ds.load()
