from raga import *
import datetime


run_name = f"ocr_missing_test_analysis-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
print(run_name)
print("*******************")
test_session = TestSession(project_name="testingProject", run_name= run_name, access_key="FZYmgBqiLqyYWQKqQjZk", secret_key="7tjSBEAxT46v0BfPUJREU4UKAuhjG7gUr0B6hVca", host="http://3.111.106.226:8080")
# test_session = TestSession(project_name="OCR", run_name= run_name, access_key="4J3dt6rGD5bq9VcvAhiL", secret_key="RPKHDiztepcrDO1kKpE7kqVm9LZ6azkjiezSzeoP", host="https://backend.prod1.ragaai.ai")


rules = OcrRules()
rules.add(expected_detection={"merchant_name": 1,"merchant_address": 1, "receipt_number": 1,"total_amount": 1,"tax_amount": 1,"date":1, "none":1})


ocr_test = ocr_missing_test_analysis(test_session=test_session,
                             dataset_name = "nano_net_dataset_1_dec_v12",
                             test_name = "ocr_missing_value",
                             model = "nanonet_model",
                             type = "ocr",
                             output_type="missing_value",
                             rules = rules)

test_session.add(ocr_test)

test_session.run()