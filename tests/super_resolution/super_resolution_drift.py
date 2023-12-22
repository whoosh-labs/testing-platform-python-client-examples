from raga import *
import datetime

run_name = f"Drift_Superresolution-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"


test_session = TestSession(project_name="testingProject", run_name = run_name, profile="dev")


rules = DriftDetectionRules()
rules.add(type="anomaly_detection", dist_metric="Mahalanobis", _class="ALL", threshold=0.6)

edge_case_detection = data_drift_detection(test_session=test_session,
                                           test_name="Drift-detection-test",
                                           dataset_name="super_resolution_data_v3",
                                           embed_col_name="imageEmbedding",
                                           output_type = "super_resolution",
                                           rules = rules)


test_session.add(edge_case_detection)
test_session.run()