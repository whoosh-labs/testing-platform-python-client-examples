from raga import *
import pandas as pd
import json
import datetime
from urllib.parse import urlsplit, urlunsplit


# https://ragatesitng-dev-storage.s3.ap-south-1.amazonaws.com/datasets/automobile/train_modelA.json
# https://ragatesitng-dev-storage.s3.ap-south-1.amazonaws.com/datasets/automobile/train_modelB.json

ds_json_file = "/home/ubuntu/developments/testing-platform-python-client/raga/examples/assets/train_modelA.json"
ds_json_file1 = "/home/ubuntu/developments/testing-platform-python-client/raga/examples/assets/train_modelA.json"

def replace_base_url(original_url):
    new_base_url = "https://ragacloudstorage.s3.ap-south-1.amazonaws.com"
    
    # Parse the original URL
    parsed_url = urlsplit(original_url)
    new_parsed_url = urlsplit(new_base_url)
    
    # Construct the new URL with the new base URL
    updated_url = urlunsplit((new_parsed_url.scheme, new_parsed_url.netloc, parsed_url.path, parsed_url.query, parsed_url.fragment))
    
    return updated_url

test_df = []
with open(ds_json_file, 'r') as json_file:
    # Load JSON data
    json_data = json.load(json_file)
    
    # Process the JSON data
    transformed_data = []
    for item in json_data:
        AnnotationsV1 = ImageDetectionObject()
        ROIVectorsM1 = ROIEmbedding()
        ImageVectorsM1 = ImageEmbedding()
        for index, detection in enumerate(item["outputs"][0]["detections"]):
            id = index+1
            AnnotationsV1.add(ObjectDetection(Id=id, ClassId=0, ClassName=detection['class'], Confidence=detection['confidence'], BBox= detection['bbox'], Format="xywh_normalized"))
            ROIVectorsM1.add(id=id, embedding_values=[float(num_str) for num_str in detection['roi_embedding']])
        
        attributes_dict = {}
        attributes = item.get("attributes", {})
        for key, value in attributes.items():
            attributes_dict[key] = StringElement(value)

        image_embeddings = item.get("image_embedding", {})
        for value in image_embeddings:
            ImageVectorsM1.add(Embedding(value))

        data_point = {
            'ImageUri': replace_base_url(item["image_url"]),
            'ImageId': item["inputs"][0],
            'TimeOfCapture': TimeStampElement(item["capture_time"]),
            'Annotations': AnnotationsV1,
            'ImageVectors': ImageVectorsM1,
            'ROIVectors': ROIVectorsM1,
        }

        merged_dict = {**data_point, **attributes_dict}

        test_df.append(merged_dict)
        

pd_data_frame = pd.DataFrame(test_df)

# data_frame_extractor(pd_data_frame).to_csv("xyz_t.csv",index=False) # converted csv file 

test_df1 = []
with open(ds_json_file1, 'r') as json_file:
    # Load JSON data
    json_data = json.load(json_file)
    
    # Process the JSON data
    transformed_data = []
    for item in json_data:
        AnnotationsV2 = ImageDetectionObject()
        ROIVectorsM2 = ROIEmbedding()
        ImageVectorsM2 = ImageEmbedding()
        for index, detection in enumerate(item["outputs"][0]["detections"]):
            id = index+1
            AnnotationsV2.add(ObjectDetection(Id=id, ClassId=0, ClassName=detection['class'], Confidence=detection['confidence'], BBox= detection['bbox'], Format="xywh_normalized"))
            ROIVectorsM2.add(id=id, embedding_values=[float(num_str) for num_str in detection['roi_embedding']])
        
        attributes_dict = {}
        attributes = item.get("attributes", {})
        for key, value in attributes.items():
            attributes_dict[key] = value

        image_embeddings = item.get("image_embedding", {})
        for value in image_embeddings:
            ImageVectorsM2.add(value)

        data_point = {
            'ImageUri':replace_base_url(item["image_url"]),
            'ImageId': item["inputs"][0],
            'TimeOfCapture': TimeStampElement(item["capture_time"]),
            'Annotations': AnnotationsV2,
            'ImageVectors': ImageVectorsM2,
            'ROIVectors': ROIVectorsM2,
        }

        merged_dict = {**data_point, **attributes_dict}

        test_df1.append(merged_dict)
        

pd_data_frame1 = pd.DataFrame(test_df1)
# pd_data_frame1.to_csv("xyz_f.csv",index=False)

model_name = "Train"
model_name1 = "Field"

#create schema object of RagaSchema instance
schema = RagaSchema()
schema.add("ImageUri", ImageUriSchemaElement())
schema.add("ImageId", PredictionSchemaElement())
schema.add("TimeOfCapture", TimeOfCaptureSchemaElement())
schema.add("Reflection", AttributeSchemaElement())
schema.add("Overlap", AttributeSchemaElement())
schema.add("CameraAngle", AttributeSchemaElement())
schema.add("Annotations", InferenceSchemaElement(model=model_name))
schema.add("ImageVectors", ImageEmbeddingSchemaElement(model=model_name))
schema.add("ROIVectors", RoiEmbeddingSchemaElement(model=model_name))

schema1 = RagaSchema()
schema1.add("ImageUri", ImageUriSchemaElement())
schema1.add("ImageId", PredictionSchemaElement())
schema1.add("TimeOfCapture", TimeOfCaptureSchemaElement())
schema1.add("Reflection", AttributeSchemaElement())
schema1.add("Overlap", AttributeSchemaElement())
schema1.add("CameraAngle", AttributeSchemaElement())
schema1.add("Annotations", InferenceSchemaElement(model=model_name1))
schema1.add("ImageVectors", ImageEmbeddingSchemaElement(model=model_name1))
schema1.add("ROIVectors", RoiEmbeddingSchemaElement(model=model_name1))

project_name = "testingProject" # Project Name
run_name = f"automobile-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}" # Experiment Name
dataset_name_t = "train-dataset-nov-29-v1" # Dataset Name for train
dataset_name_f = "field-dataset-nov-29-v1" # Dataset Name for feild


test_session = TestSession(project_name=project_name,run_name=run_name, profile="raga-dev-new")



cred = DatasetCreds(region="ap-south-1")

test_ds1 = Dataset(test_session=test_session,
                  name=dataset_name_t,
                  type=DATASET_TYPE.IMAGE,
                  data=pd_data_frame,
                  schema=schema,
                  creds=cred)

#load schema and pandas data frame of train dataset

test_ds1.load()



#create test_ds object of Feild Dataset instance
test_ds2 = Dataset(test_session=test_session,
                  name=dataset_name_f,
                  type=DATASET_TYPE.IMAGE,
                  data=pd_data_frame,
                  schema=schema,
                  creds=cred)

#load schema and pandas data frame of Feild Dataset
test_ds2.load()

