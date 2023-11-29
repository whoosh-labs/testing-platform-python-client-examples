from raga import *
import datetime

run_name = f"data_drift_test-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", run_name= run_name, profile="raga-dev-new")

train_dataset_name = "train-dataset-nov-29-v1"
field_dataset_name = "field-dataset-nov-29-v1"
train_embed_col_name = "ImageEmbedding"
field_embed_col_name = "ImageEmbedding"
level = "image"

# train_embed_col_name = "ROIVectors Train"      #for train col in roi level
# field_embed_col_name = "ROIVectors Field"      #for field col in roi level
# level = "roi")


aggregation_level = AggregationLevelElement()
# aggregation_level.add("Reflection")
# aggregation_level.add("Overlap")
# aggregation_level.add("CameraAngle")

rules = DriftDetectionRules()
rules.add(type="anomaly_detection", dist_metric="Euclidian", _class = "ALL", threshold = "0.5")

edge_case_detection = data_drift_detection(test_session, 
                                           testName="data_drift", 
                                           train_dataset_name=train_dataset_name, field_dataset_name=field_dataset_name, train_embed_col_name=train_embed_col_name, 
                                           field_embed_col_name = field_embed_col_name , 
                                           level = level, 
                                           aggregation_level=aggregation_level, 
                                           rules = rules)

# #add payload into test_session object
test_session.add(edge_case_detection)

# #run added ab test model payload
test_session.run()
