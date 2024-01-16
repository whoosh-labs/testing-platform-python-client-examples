import pathlib
from raga import *
import pandas as pd
import datetime

weather_data = {
    "overcast": "overcast",
    "fog": "fog",
    "cloudy": "overcast",
    "sunny": "clear",
    "clear": "clear",
    "snow": "snow",
    "rain": "rain",
    "partly cloudy": "overcast",
    "light rain": "rain",
    "patchy light rain with thunder": "rain",
    "patchy light rain": "rain",
    "light drizzle": "rain",
    "light freezing rain": "rain",
    "thundery outbreaks possible": "overcast",
    "patchy rain possible": "overcast",
    "patchy light drizzle": "rain",
    "heavy freezing drizzle":"rain",
    "moderate snow": "snow",
    "moderate rain at times": "rain",
    "moderate rain": "rain",
    "moderate or heavy snow showers": "snow",
    "moderate or heavy rain shower": "rain",
    "moderate or heavy rain with thunder":"rain",
    "moderate or heavy freezing rain":"rain",
    "heavy rain":"rain",
    "heavy snow":"snow",
    "mist": "fog",
    "freezing fog":"fog",
    "light snow": "snow",
    "light snow showers": "snow",
    "light sleet showers": "snow",
    "light sleet": "snow",
    "light rain shower": "rain",
    "blowing snow": "snow",
    "patchy light snow":"snow",
    "patchy moderate snow":"snow",
    "ice pellets":"snow",
    "blizzard":"snow",
    "":"NA"
}

def get_timestamp_x_hours_ago(hours):
    current_time = datetime.datetime.now()
    delta = datetime.timedelta(days=90, hours=hours)
    past_time = current_time - delta
    timestamp = int(past_time.timestamp())
    return timestamp

def img_url(x):
    return f"https://ragacloudstorage.s3.ap-south-1.amazonaws.com/1/StopSign_Part1_event.json/data_points/{pathlib.Path(x).stem}/{x}"

def json_parser(event_1):
    event_1_df = pd.read_json(event_1)

    event_1_df_exploded = event_1_df.explode('inputs')

    attributes = event_1_df["attributes"].apply(pd.Series)

    event_1_df_exploded = pd.concat([event_1_df_exploded, attributes], axis=1)
    data_frame = pd.DataFrame()
    data_frame["videoId"] = event_1_df_exploded["inputs"].apply(lambda x: pathlib.Path(x).stem)
    data_frame["videoUrl"] = event_1_df_exploded["inputs"].apply(lambda x: img_url(x))
    data_frame["timeOfCapture"] = event_1_df_exploded.apply(lambda row: TimeStampElement(get_timestamp_x_hours_ago(row.name)), axis=1)
    data_frame["dutyType"] = event_1_df_exploded["dutyType"].apply(lambda x: x)
    data_frame["time_of_day"] = event_1_df_exploded["time_of_day"].apply(lambda x: x)
    data_frame["weather"] = event_1_df_exploded["weather"].apply(lambda x: weather_data[str(x).lower()])
    data_frame["scene"] = event_1_df_exploded["scene"].apply(lambda x: x)
    data_frame["tags"] = event_1_df_exploded["tags"].apply(lambda x: x)    
    return data_frame.head(5)

####################################################################
## You can use csv url or download the file and use the file path ##
####################################################################

pd_video_data_frame = json_parser("https://ragatesitng-dev-storage.s3.ap-south-1.amazonaws.com/datasets/lightmetrics/lightmetrics_stop_sign_dataset.json")

########
## OR ##
########
# pd_video_data_frame = json_parser("./assets/Complex-America-Stop-Event.json")


schema = RagaSchema()
schema.add("videoId", PredictionSchemaElement())
schema.add("videoUrl", ImageUriSchemaElement())
schema.add("timeOfCapture", TimeOfCaptureSchemaElement())
schema.add("dutyType", AttributeSchemaElement())
schema.add("time_of_day", AttributeSchemaElement())
schema.add("weather", AttributeSchemaElement())
schema.add("scene", AttributeSchemaElement())
schema.add("tags", AttributeSchemaElement())

# # create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", profile="dev")


creds = DatasetCreds(region="ap-south-1")

# #create test_ds object of Dataset instance
video_ds = Dataset(test_session=test_session,
                  name="test-lm-dataset",
                  type=DATASET_TYPE.VIDEO,
                  data=pd_video_data_frame,
                  schema=schema,
                  creds=creds)
#load schema and pandas data frame
video_ds.lightmetrics_data_upload()