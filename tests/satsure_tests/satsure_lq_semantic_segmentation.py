from raga import *
import datetime

run_name = f"satsure-test-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", run_name= run_name, profile="raga-dev-new")


dataset_name = "Enter_your_dataset_name"

rules = LQRules()
rules.add(metric="mistake_score", label=["ALL"], metric_threshold=0.065)

edge_case_detection = labelling_quality_test(test_session=test_session,
                                            dataset_name = dataset_name,
                                            test_name = "Labeling Quality Test",
                                            type = "labelling_consistency",
                                            output_type="semantic_segmentation",
                                            mistake_score_col_name = "MistakeScores",
                                            rules = rules)
test_session.add(edge_case_detection)

test_session.run()