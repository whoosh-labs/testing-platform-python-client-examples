from raga import *
import datetime

run_name = f"data_leakage_test-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", run_name= run_name, profile="dev1")

rules = LQRules()
rules.add(metric = 'overlapping_samples', metric_threshold = 0.9)


train_dataset_name = "Enter_train_dataset_name"
field_dataset_name = "Enter_field_dataset_name"


edge_case_detection = data_leakage_test(test_session=test_session,
                                           test_name="Data-Leakage-Test",
                                           train_dataset_name=train_dataset_name,
                                           dataset_name=field_dataset_name,
                                           type = "data_leakage",
                                           output_type="image_data",
                                           train_embed_col_name="embedding",
                                           embed_col_name = "embedding",
                                           rules = rules)

test_session.add(edge_case_detection)

test_session.run()
