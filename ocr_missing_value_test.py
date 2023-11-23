from raga import *
import datetime


run_name = f"ocr_missing_test_analysis-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
# print(run_name)
# print("*******************")
test_session = TestSession(project_name="testingProject", run_name= run_name, access_key="75TS1h6fyP1SVBrtuOFd", secret_key="AIGc5XextI7s8RnWNbzPby9azQkmH374gcZ1eWvX", host="http://3.111.106.226:8080")


rules = OcrRules()
rules.add(expected_detection={"merchant_name": 1,"date": 1,"total":1})



ocr_test = ocr_missing_test_analysis(test_session=test_session,
                             dataset_name = "nano_net_dataset_14_nov_v1",
                             test_name = "ocr_missing_value",
                             model = "nanonet_model",
                             type = "ocr",
                             output_type="missing_value",
                             rules = rules)

test_session.add(ocr_test)

test_session.run()