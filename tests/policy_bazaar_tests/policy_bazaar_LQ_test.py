from raga import *
import datetime

run_name = f"policy_bazaar-test-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", run_name= run_name, profile="raga-dev-new")

rules = LQRules()
rules.add(metric="mistake_score", label=["ALL"])

dataset_name = "policy_bazaar_train_dataset"

edge_case_detection = labelling_quality_test(test_session=test_session,
                                             dataset_name = dataset_name,
                                             test_name = "pb_labelling_quality_2",
                                             train_model_column_name = "target",
                                             field_model_column_name = "target",
                                             type = "labelling_consistency",
                                             output_type="embedding_data",
                                             embedding_train_col_name = "embedding",
                                             embedding_field_col_name = "embedding",
                                             rules = rules)
# print(edge_case_detection)
test_session.add(edge_case_detection)

test_session.run()
