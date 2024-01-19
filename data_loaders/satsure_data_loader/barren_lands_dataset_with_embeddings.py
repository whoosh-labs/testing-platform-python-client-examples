import pathlib
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


def image_url(x):
    return f"https://raga-dev-testing-platform-backend-s3-storage.s3.ap-south-1.amazonaws.com/1/BarrenLands_image/{pathlib.Path(x).name}"

def mask_url(x):
    return f"https://raga-dev-testing-platform-backend-s3-storage.s3.ap-south-1.amazonaws.com/1/BarrenLands/{pathlib.Path(x).name}"

def csv_parser(file_path):
    df = pd.read_csv(file_path)        
    df["ImageUri"] = df["SourceLink"].apply(lambda x: StringElement(image_url(x)))
    df["Annotations"] = df['Annotations'].apply(lambda x:StringElement(mask_url(x)))
    return df.head(10)

####################################################################
## You can use csv url or download the file and use the file path ##
####################################################################

data_frame = csv_parser("https://ragatesitng-dev-storage.s3.ap-south-1.amazonaws.com/datasets/satsure/barren_lands_dataset_with_embeddings.csv")

########
## OR ##
########

# data_frame = csv_parser("./barren_lands_dataset_with_embeddings.csv")


schema = RagaSchema()
schema.add("ImageId", PredictionSchemaElement())
schema.add("ImageUri", ImageUriSchemaElement())
schema.add("TimeOfCapture", TimeOfCaptureSchemaElement())
schema.add("SourceLink", FeatureSchemaElement())
schema.add("Annotations", TIFFSchemaElement(label_mapping=label_to_classname, schema="tiff"))
schema.add("ImageEmbedding", ImageEmbeddingSchemaElement(model="Satsure Embedding Model"))

test_session = TestSession(project_name="testingProject", profile="dev")

cred = DatasetCreds(region="ap-south-1")

#create test_ds object of Dataset instance
test_ds = Dataset(test_session=test_session, 
                  name="BarrenLands-final-15-jan-v1", 
                  type=DATASET_TYPE.IMAGE,
                  data=data_frame, 
                  schema=schema, 
                  creds=cred)

# #load schema and pandas data frame
# test_ds.load()  
test_ds.head()

