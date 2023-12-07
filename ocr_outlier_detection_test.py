from raga import *
import datetime


run_name = f"ocr_anomaly_test_analysis-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

# test_session = TestSession(project_name="testingProject", run_name= run_name, access_key="864yWXlrWG1AqAtpH7GG", secret_key="dKD6OEpc8ZlaG6tDVRbP1aDYWx7P8DvHMp94mHeS", host="http://3.111.106.226:8080")
test_session = TestSession(project_name="OCR", run_name= run_name, access_key="Xiv7xvvdlEJ8mMeqmqgW", secret_key="rxC4ZxZrgvVk7UxqSWeqkuEHoSzqayELCihCGms1", host="https://backend.prod1.ragaai.ai")


rules = OcrAnomalyRules()
rules.add(type="anomaly_detection", dist_metric="DistanceMetric", threshold=0.2)



ocr_test = ocr_anomaly_test_analysis(test_session=test_session,
                             dataset_name = "nano_net_dataset_december_demo",
                             test_name = "ocr_anomaly_detection",
                             model = "nanonet_model",
                             type = "ocr",
                             output_type="anomaly_detection",
                             rules = rules)

test_session.add(ocr_test)

test_session.run()