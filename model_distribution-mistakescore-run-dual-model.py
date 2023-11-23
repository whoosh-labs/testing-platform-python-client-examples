import pathlib
from raga import *
import pandas as pd

def image_url(x):
    return f"https://ragatesitng-dev-storage.s3.ap-south-1.amazonaws.com/1/satsure_rgb/{x}"

def csv_parser(file_path):
    df = pd.read_csv(file_path)
    data_frame = pd.DataFrame()
    data_frame["ImageId"] = df["filename"].apply(lambda x: f"{x}.tif")
    data_frame["ImageUri"] = df["filename"].apply(lambda x: image_url(f"{x}.tif"))
    return data_frame.head(50)

data_frame = csv_parser("./assets/satsure.csv")


schema = RagaSchema()
schema.add("ImageId", PredictionSchemaElement())
schema.add("ImageUri", ImageUriSchemaElement())

test_session = TestSession(project_name="testingProject", 
                           run_name= "Embedding Generator Run Test", 
                           profile="nanonets-uat")


# #create test_ds object of Dataset instance
# test_ds = Dataset(test_session=test_session, 
#                   name="model-distribution-test-nov16-v1", 
#                   type=DATASET_TYPE.IMAGE,
#                   data=data_frame, 
#                   schema=schema)

# # #load schema and pandas data frame
# test_ds.load()

# model_exe_fun = ModelExecutorFactory().get_model_executor(test_session=test_session, 
#                                                           model_name="Satsure Embedding Model", 
#                                                           version="0.1.1", wheel_path="/home/ubuntu/developments/testing-platform-python-client/raga/examples/assets/satsure/raga_models-0.1.1-cp311-cp311-linux_x86_64.whl")

# df = model_exe_fun.execute(init_args={"device": "cpu"}, 
#                            execution_args={"input_columns":{"img_paths":"ImageUri"}, 
#                                            "output_columns":{"embedding":"ImageEmbedding"},
#                                            "column_schemas":{"embedding":ImageEmbeddingSchemaElement(model="Satsure Embedding Model")}}, 
#                            data_frame=test_ds)

# test_ds.load()

# model_exe_fun = ModelExecutorFactory().get_model_executor(test_session=test_session, 
#                                                           model_name="Satsure Embedding Model", 
#                                                           version="0.1.1", wheel_path="/home/ubuntu/developments/Annotation-Consistency-Package/dist/raga_models-0.1.7.32-py3-none-any.whl")

# df = model_exe_fun.execute(init_args={"device": "cpu", 
#                                       "image_folders":["/home/ubuntu/developments/datasets/dataset/small_rgb"], 
#                                       "annotation_folders":["/home/ubuntu/developments/datasets/dataset/small_lulc"]}, 
#                            execution_args={"input_columns":{"img_paths":"ImageUri"}, 
#                                            "output_columns":{"mistake_score":"MistakeScore"},
#                                            "column_schemas":{"mistake_score":ImageEmbeddingSchemaElement(model="Satsure Embedding Model")}}, 
#                            data_frame=test_ds)

# test_ds.load()


# model_exe_fun = ModelExecutorFactory().get_model_executor(test_session=test_session, 
#                                                           model_name="Satsure Embedding Model", 
#                                                           version="0.1.1", wheel_path="/home/ubuntu/developments/testing-platform-python-client/raga/examples/assets/satsure/raga_models-0.1.1-cp311-cp311-linux_x86_64.whl")

# df = model_exe_fun.execute(init_args={"device": "cpu"}, 
#                            execution_args={"input_columns":{"img_paths":"ImageUri"}, 
#                                            "output_columns":{"embedding":"ImageEmbedding"},
#                                            "column_schemas":{"embedding":ImageEmbeddingSchemaElement(model="Satsure Embedding Model")}}, 
#                            data_frame=test_ds)

# test_ds.load()

# model_exe_fun = ModelExecutorFactory().get_model_executor(test_session=test_session, 
#                                                           model_name="Satsure Embedding Model", 
#                                                           version="0.1.1", wheel_path="/home/ubuntu/developments/Annotation-Consistency-Package/dist/raga_models-0.1.7.32-py3-none-any.whl")

# df = model_exe_fun.execute(init_args={"device": "cpu", 
#                                       "image_folders":["/home/ubuntu/developments/datasets/dataset/small_rgb"], 
#                                       "annotation_folders":["/home/ubuntu/developments/datasets/dataset/small_lulc"]}, 
#                            execution_args={"input_columns":{"img_paths":"ImageUri"}, 
#                                            "output_columns":{"mistake_score":"MistakeScore"},
#                                            "column_schemas":{"mistake_score":ImageEmbeddingSchemaElement(model="Satsure Embedding Model")}}, 
#                            data_frame=test_ds)

# test_ds.load()