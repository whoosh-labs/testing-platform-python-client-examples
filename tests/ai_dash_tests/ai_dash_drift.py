from raga import *
import datetime

run_name = f"Ai_dash_Test-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", run_name= run_name, profile="raga-dev-new")

rules = DriftDetectionRules()
rules.add(type="single_class_anomaly_detection", dist_metric="Mahalanobis", _class="ALL", threshold=21.3)

train_dataset_name="Enetr_your_train_dataset_name"
field_dataset_name="Enetr_your_field_dataset_name"

#To Run OD Test
edge_case_detection = data_drift_detection(test_session=test_session,
                                           test_name="Drift-detection-test",
                                           train_dataset_name=train_dataset_name,
                                           field_dataset_name=field_dataset_name,
                                           train_embed_col_name="ImageVectorsM1",
                                           field_embed_col_name = "ImageVectorsM1",
                                           output_type = "object_detection",
                                           level = "roi",
                                           rules = rules)


test_session.add(edge_case_detection)

test_session.run()