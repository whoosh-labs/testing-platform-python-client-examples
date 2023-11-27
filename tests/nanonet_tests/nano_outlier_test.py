from raga import *
import datetime

run_name = f"Nano_Net_Test-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", run_name= run_name, profile="raga-dev-new")


rules = OcrAnomalyRules()
rules.add(type="anomaly_detection", dist_metric="DistanceMetric", threshold=0.2)

dataset_name = "Enter_your_loaded_dataset_name"

edge_case_detection = ocr_anomaly_test_analysis(test_session=test_session,
                                                dataset_name = dataset_name,
                                                test_name = "ocr_anomaly_detection",
                                                model = "nanonet_model",
                                                type = "ocr",
                                                output_type="anomaly_detection",
                                                rules = rules)

test_session.add(edge_case_detection)

test_session.run()