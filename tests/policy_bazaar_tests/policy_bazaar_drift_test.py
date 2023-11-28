from raga import *
import datetime

run_name = f"policy_bazaar-test-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", run_name= run_name, profile="raga-dev-new")

rules = DriftDetectionRules()
rules.add(type="anomaly_detection", dist_metric="Mahalanobis", _class="ALL", threshold=45)

train_dataset_name = "Enter_train_dataset_name"
field_dataset_name = "Enter_field_dataset_name"

edge_case_detection = data_drift_detection(test_session=test_session,
                                           test_name="Drift-detection-test",
                                           train_dataset_name=train_dataset_name,
                                           field_dataset_name=field_dataset_name,
                                           train_embed_col_name="embedding",
                                           field_embed_col_name = "embedding",
                                           output_type = "embedding_data",
                                           level = "image",
                                           rules = rules)

test_session.add(edge_case_detection)

test_session.run()
