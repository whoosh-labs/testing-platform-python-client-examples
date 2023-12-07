from raga import *
import datetime

run_name = f"Active_Learning-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
# print(run_name)
# print("**********")

# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", run_name= run_name, profile="dev")


dataset_name = "Enter-your-dataset-name"
budget = 10000

edge_case_detection = active_learning(test_session=test_session,
                                      dataset_name = dataset_name,
                                      test_name = "active_learning_5",
                                      type = "active_learning",
                                      output_type="curated_dataset",
                                      embed_col_name="hr_embedding",
                                      budget=budget)

test_session.add(edge_case_detection)

test_session.run()