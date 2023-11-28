from raga import *
import datetime
from raga.test_config import labelling_quality_test
run_name = f"poc-test-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", run_name= run_name, profile="dev")

dataset_name = "policy_bazaar_train_dataset"

rules = LQRules()
rules.add(metric="mistake_score", label=["ALL"])

edge_case_detection = labelling_quality_test(test_session=test_session,
                                             dataset_name = dataset_name,
                                             test_name = "pb_labelling_quality_2",
                                             train_model_col_name = "target",
                                             field_model_col_name= "target",
                                             type = "labelling_consistency",
                                             output_type="embedding_data",
                                             embedding_train_col_name = "embedding",
                                             embedding_field_col_name = "embedding",
                                             rules = rules)

# edge_case_detection = labelling_quality_test_PB(test_session=test_session,
#                                              dataset_name = dataset_name,
#                                              test_name = "pb_labelling_quality_2",
#                                              trainModelColumnName = "target",
#                                              fieldModelColumnName = "target",
#                                              type = "labelling_consistency",
#                                              output_type="embedding_data",
#                                              embeddingTrainColName = "embedding",
#                                              embeddingFieldColName = "embedding",
#                                              rules = rules)

print(edge_case_detection)
# test_session.add(edge_case_detection)

# test_session.run()



