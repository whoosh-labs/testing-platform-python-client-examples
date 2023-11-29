from raga import *
import datetime

run_name = f"Policy-Bazaar-labeling-quality-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
# print(run_name)
# print("******************")
test_session = TestSession(project_name="testingProject", run_name= run_name, access_key="y2EkSDP1yryEpHAzuJO8", secret_key="DR7ZZt22LVe0iIJ1psG2beegUmaQ53vTUhpX5Fg4", host="http://3.111.106.226:8080")

rules = LQRules()
rules.add(metric="mistake_score", label=["ALL"])


edge_case_detection = labelling_quality_test(test_session=test_session,
                                             dataset_name = "labelling_quality_train_PB_v1",
                                             test_name = "pb_labelling_quality_2",
                                             train_model_column_name = "target",
                                             field_model_column_name = "target",
                                             type = "labelling_consistency",
                                             output_type="embedding_data",
                                             embedding_train_col_name = "embedding",
                                             embedding_field_col_name = "embedding",
                                             rules = rules)
test_session.add(edge_case_detection)

test_session.run()
