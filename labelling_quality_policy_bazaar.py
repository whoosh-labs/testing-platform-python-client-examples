from raga import *
import datetime

run_name = f"Policy-Bazaar-labeling-quality-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
# print(run_name)
# print("******************")
test_session = TestSession(project_name="testingProject", run_name= run_name, access_key="y2EkSDP1yryEpHAzuJO8", secret_key="DR7ZZt22LVe0iIJ1psG2beegUmaQ53vTUhpX5Fg4", host="http://3.111.106.226:8080")

rules = LQRules()
rules.add(metric="mistake_score", label=["ALL"], metric_threshold=0.065)


edge_case_detection = labelling_quality_test_PB(test_session=test_session,
                                             dataset_name = "labelling_quality_train_PB_v1",
                                             test_name = "pb_labelling_quality_2",
                                             trainModelColumnName = "target",
                                             fieldModelColumnName = "target",
                                             type = "labelling_consistency",
                                             output_type="embedding_data",
                                             embeddingTrainColName = "imageEmbedding_1",
                                             embeddingFieldColName = "imageEmbedding_1",
                                             rules = rules)
test_session.add(edge_case_detection)

test_session.run()