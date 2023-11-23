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

def csv_parser(csv_path):
    data_frame = pd.read_csv(csv_path).head(10)
    return data_frame


pd_data_frame = csv_parser("./assets/test.csv")


schema = RagaSchema()
schema.add("ImageId", PredictionSchemaElement())
schema.add("MistakeScores", MistakeScoreSchemaElement(ref_col_name="Annotations"))


run_name = f"loader_satsure_new_ready_load-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

# satsure-prod
test_session = TestSession(project_name="testingProject", run_name = run_name, profile="dev")

cred = DatasetCreds(region="us-east-2")

#create test_ds object of Dataset instance
test_ds = Dataset(test_session=test_session,
                  name="satsure_new_data_nov",
                  type=DATASET_TYPE.IMAGE,
                  data=pd_data_frame,
                  schema=schema,
                  creds=cred,
                  temp=True)

#load to server
test_ds.load()