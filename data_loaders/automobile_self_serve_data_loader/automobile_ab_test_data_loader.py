from raga import *
import pandas as pd
import json
import datetime
from urllib.parse import urlsplit, urlunsplit

def get_timestamp_x_hours_ago(hours):
    current_time = datetime.datetime.now()
    delta = datetime.timedelta(days=90, hours=hours)
    past_time = current_time - delta
    timestamp = int(past_time.timestamp())
    return timestamp

def replace_base_url(original_url):
    new_base_url = "https://ragacloudstorage.s3.ap-south-1.amazonaws.com"
    
    # Parse the original URL
    parsed_url = urlsplit(original_url)
    new_parsed_url = urlsplit(new_base_url)
    
    # Construct the new URL with the new base URL
    updated_url = urlunsplit((new_parsed_url.scheme, new_parsed_url.netloc, parsed_url.path, parsed_url.query, parsed_url.fragment))
    
    return updated_url

def convert_json_to_data_frame(json_file_path_model_1, json_file_path_model_2, json_file_path_model_3):
    test_data_frame = []
    with open(json_file_path_model_1, 'r') as json_file:
        # Load JSON data
        model_1 = json.load(json_file)
    with open(json_file_path_model_2, 'r') as json_file:
        # Load JSON data
        model_2 = json.load(json_file)
    
    with open(json_file_path_model_3, 'r') as json_file:
        # Load JSON data
        model_gt = json.load(json_file)

    # Create a dictionary to store the inputs and corresponding data points
    inputs_dict = {}
    hr = 1
    # Process model_1 data
    for item in model_1:
        inputs = item["inputs"]
        inputs_dict[tuple(inputs)] = item
    
    # Process model_2 data
    for item in model_2:
        inputs = item["inputs"]
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
                attributes_dict[key] = value
            image_embeddings = item.get("image_embedding", {})
            for value in image_embeddings:
                ImageVectorsM1.add(float(value))

        merged_item = inputs_dict.get(tuple(inputs), {})
        AnnotationsV2 = ImageDetectionObject()
        ROIVectorsM2 = ROIEmbedding()
        ImageVectorsM2 = ImageEmbedding()
        for index, detection in enumerate(merged_item["outputs"][0]["detections"]):
            id = index+1
            AnnotationsV2.add(ObjectDetection(Id=id, ClassId=0, ClassName=detection['class'], Confidence=detection['confidence'], BBox= detection['bbox'], Format="xywh_normalized"))
            ROIVectorsM2.add(id=id, embedding_values=[float(num_str) for num_str in detection['roi_embedding']])
        
        image_embeddings = merged_item.get("image_embedding")
        for value in image_embeddings:
            ImageVectorsM2.add(float(value))
        
        merged_item2 = inputs_dict.get(tuple(inputs))
        AnnotationsV3 = ImageDetectionObject()
        ROIVectorsM3 = ROIEmbedding()
        ImageVectorsM3 = ImageEmbedding()
        for index, detection in enumerate(merged_item2["outputs"][0]["detections"]):
            id = index+1
            AnnotationsV3.add(ObjectDetection(Id=id, ClassId=0, ClassName=detection['class'], Confidence=detection['confidence'], BBox= detection['bbox'], Format="xywh_normalized"))
            ROIVectorsM3.add(id=id, embedding_values=[float(num_str) for num_str in detection['roi_embedding']])
        
        image_embeddingsGT = merged_item2.get("image_embedding")
        for value in image_embeddingsGT:
            ImageVectorsM3.add(float(value))


        data_point = {
            'ImageUri':replace_base_url(item["image_url"]),
            'ImageId': item["inputs"][0],
            'TimeOfCapture': TimeStampElement(get_timestamp_x_hours_ago(hr)),
            'AnnotationsV1': AnnotationsV1,
            'ROIVectorsM1': ROIVectorsM1,
            'ImageVectorsM1': ImageVectorsM1,
            'AnnotationsV2': AnnotationsV2,
            'ROIVectorsM2': ROIVectorsM2,
            'ImageVectorsM2': ImageVectorsM2,
            'AnnotationsV3': AnnotationsV3,
            'ROIVectorsM3': ROIVectorsM3,
            'ImageVectorsM3': ImageVectorsM3,
            
        }

        merged_dict = {**data_point, **attributes_dict}
        test_data_frame.append(merged_dict)
        hr+=1

    return test_data_frame


# Please download JSON files
# https://ragatesitng-dev-storage.s3.ap-south-1.amazonaws.com/datasets/automobile/gt.json
# https://ragatesitng-dev-storage.s3.ap-south-1.amazonaws.com/datasets/automobile/ma.json
# https://ragatesitng-dev-storage.s3.ap-south-1.amazonaws.com/datasets/automobile/mb.json


#Convert JSON dataset to pandas Data Frame
pd_data_frame = pd.DataFrame(convert_json_to_data_frame("/home/ubuntu/developments/testing-platform-python-client/raga/examples/assets/ma.json", "/home/ubuntu/developments/testing-platform-python-client/raga/examples/assets/mb.json", "/home/ubuntu/developments/testing-platform-python-client/raga/examples/assets/gt.json"))
#pd_data_frame.to_pickle("TestingDataFrame.pkl")

schema = RagaSchema()
schema.add("ImageUri", ImageUriSchemaElement())
schema.add("ImageId", PredictionSchemaElement())
schema.add("TimeOfCapture", TimeOfCaptureSchemaElement())
schema.add("Reflection", AttributeSchemaElement())
schema.add("Overlap", AttributeSchemaElement())
schema.add("CameraAngle", AttributeSchemaElement())
schema.add("AnnotationsV1", InferenceSchemaElement(model="modelA"))
schema.add("ImageVectorsM1", ImageEmbeddingSchemaElement(model="modelA"))
schema.add("ROIVectorsM1", RoiEmbeddingSchemaElement(model="modelA"))
schema.add("AnnotationsV2", InferenceSchemaElement(model="modelB"))
schema.add("ImageVectorsM2", ImageEmbeddingSchemaElement(model="modelB"))
schema.add("ROIVectorsM2", RoiEmbeddingSchemaElement(model="modelB"))
schema.add("AnnotationsV3", InferenceSchemaElement(model="modelGT"))
schema.add("ImageVectorsM3", ImageEmbeddingSchemaElement(model="modelGT"))
schema.add("ROIVectorsM3", RoiEmbeddingSchemaElement(model="modelGT"))


project_name = "testingProject" # Project Name
run_name= f"automobile-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"  # Experiment Name should always be unique
dataset_name = "automobile-dataset" # Dataset Name


#create test_session object of TestSession instance
test_session = TestSession(project_name=project_name, run_name=run_name, profile="raga-dev-new")

#create test_ds object of Dataset instance

cred = DatasetCreds(region="ap-south-1")

test_ds = Dataset(test_session=test_session,
                  name=dataset_name,
                  type=DATASET_TYPE.IMAGE,
                  data=pd_data_frame,
                  schema=schema,
                  creds=cred)
#load schema and pandas data frame
test_ds.load()