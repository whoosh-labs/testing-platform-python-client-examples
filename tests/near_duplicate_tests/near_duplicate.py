from raga import *
import datetime

run_name = f"Near-Duplicate-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
# print(run_name)
# print("**********")
# create test_session object of TestSession instance
# test_session = TestSession(project_name="testingProject", run_name= run_name, profile="dev")
test_session = TestSession(project_name="testingProject", run_name= run_name, profile="dev1")

rules = LQRules()
rules.add(metric="similarity_score", metric_threshold=0.99)


dataset_name = "Enter-your-dataset-name"

edge_case_detection = nearest_duplicate(test_session=test_session,
                                          dataset_name = dataset_name,
                                          test_name = "near_duplicate_detection_1",
                                          type = "nearest_neighbour",
                                          output_type="near_duplicates",
                                          embed_col_name="embedding",
                                          rules=rules)

test_session.add(edge_case_detection)

test_session.run()
