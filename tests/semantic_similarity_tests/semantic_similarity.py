from raga import *
import datetime

run_name = f"Semantic_Similarity-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
# print(run_name)
# print("**********")

# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", run_name= run_name, profile="dev1")

rules = LQRules()
rules.add(metric="similarity_score", metric_threshold=0.2)

dataset_name = "Enter-your-dataset-name"

edge_case_detection = semantic_similarity(test_session=test_session,
                                      dataset_name = dataset_name,
                                      test_name = "active_learning_1",
                                      type = "semantic_similarity",
                                      output_type="super_resolution",
                                      embed_col_name="lr_embedding",
                                      generated_embed_col_name="hr_embedding",
                                      rules=rules)

test_session.add(edge_case_detection)

test_session.run()
