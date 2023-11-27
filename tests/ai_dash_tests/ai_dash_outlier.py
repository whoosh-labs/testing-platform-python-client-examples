from raga import *
import datetime

run_name = f"Ai_dash_Test-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", run_name= run_name, profile="raga-dev-new")

dataset_name = "Enter_your_dataset_name"

rules = DriftDetectionRules()
rules.add(type="anomaly_detection", dist_metric="Mahalanobis", _class="ALL", threshold=21.3)
edge_case_detection = data_drift_detection(test_session=test_session,
                                           test_name="Drift-detection-test_AI_dash",
                                           dataset_name=dataset_name,
                                           embed_col_name = "ImageVectorsM1",
                                           output_type = "outlier_detection",
                                           rules = rules)
test_session.add(edge_case_detection)

test_session.run()