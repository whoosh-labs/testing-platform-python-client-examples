from raga import *
import datetime


run_name = f"ocr_anomaly_test_analysis-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

test_session = TestSession(project_name="testingProject", run_name= run_name, access_key="75TS1h6fyP1SVBrtuOFd", secret_key="AIGc5XextI7s8RnWNbzPby9azQkmH374gcZ1eWvX", host="http://3.111.106.226:8080")


rules = OcrAnomalyRules()
rules.add(type="anomaly_detection", dist_metric="DistanceMetric", threshold=0.2)



ocr_test = ocr_anomaly_test_analysis(test_session=test_session,
                             dataset_name = "nano_net_dataset_14_nov_v6",
                             test_name = "ocr_anomaly_detection",
                             model = "nanonet_model",
                             type = "ocr",
                             output_type="anomaly_detection",
                             rules = rules)

test_session.add(ocr_test)

test_session.run()