from raga import *
import datetime

run_name = f"LQ-IC-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

test_session = TestSession(project_name="testingProject", run_name= run_name, profile="raga-dev-new")

rules = LQRules()
rules.add(metric="mistake_score", label=["ALL"], metric_threshold=0.005)

dataset_name = "Enter-dataset-name"
edge_case_detection = labelling_quality_test(test_session=test_session,
                                            dataset_name = dataset_name,
                                            test_name = "Labeling Quality Test",
                                            type = "labelling_consistency",
                                            output_type="image_classification",
                                            mistake_score_col_name = "MistakeScore",
                                            embedding_col_name = "ImageVectorsM1",
                                            rules = rules)
test_session.add(edge_case_detection)

test_session.run()