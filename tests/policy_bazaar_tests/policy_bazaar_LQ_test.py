from raga import *
import datetime

run_name = f"policy_bazaar-test-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", run_name= run_name, profile="raga-dev-new")

rules = LQRules()
rules.add(metric="mistake_score", label=["ALL"])

dataset_name = "Enter_dataset_name"

edge_case_detection = labelling_quality_test_PB(test_session=test_session,
                                             dataset_name = dataset_name,
                                             test_name = "pb_labelling_quality_2",
                                             trainModelColumnName = "target",
                                             fieldModelColumnName = "target",
                                             type = "labelling_consistency",
                                             output_type="embedding_data",
                                             embeddingTrainColName = "embedding",
                                             embeddingFieldColName = "embedding",
                                             rules = rules)
test_session.add(edge_case_detection)

test_session.run()
