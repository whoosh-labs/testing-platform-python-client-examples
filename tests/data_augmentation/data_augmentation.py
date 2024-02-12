from raga import *
import datetime

run_name = f"data_augmentation_test-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
# print(run_name)
# print("***********")
# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", run_name= run_name, profile="raga-dev-new")

rules = DARules()
# rules.add(technique= "Zoom", scale = 2.0, Stage = "A")
# rules.add(technique= "RandomNoise", type="GaussianNoise", variance_limit = (5.0, 25.0), mean = 0.0, p = 0.5, Stage="B")
# rules.add(technique = "Flip", type="HorizontalFlip", Stage="C")

# edge_case_detection = data_augmentation_test(test_session=test_session,
#                                         test_name="Data-Augmentation-Test",
#                                         dataset_name=dataset_name,
#                                         type = "data_augmentation",
#                                         sub_type = "basic",
#                                         output_type="image_data",
#                                         rules = rules)

rules.add(default = "False", technique="Overlay", scenario_type="Rain", raindropLength="0.5", raindropWidth="0.5", raindropCount="0.5", blurSize="0.5", rainLevel="0.5", intensity="0.5", Stage="A")
#rules.add(default = "True", technique="Overlay", scenario_type="Rain", Stage="A")

dataset_name = "Enter_dataset_name"


edge_case_detection = data_augmentation_test(test_session=test_session,
                                        test_name="Data-Augmentation-Test",
                                        dataset_name=dataset_name,
                                        type = "data_augmentation",
                                        sub_type = "intermediate",
                                        output_type="image_data",
                                        rules = rules)

test_session.add(edge_case_detection)

test_session.run()
