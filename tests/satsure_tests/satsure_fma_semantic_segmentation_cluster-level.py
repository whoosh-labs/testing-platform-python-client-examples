from raga import *
import datetime

run_name = f"satsure-test-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", run_name= run_name, profile="raga-dev-new")

dataset_name = "Enter_your_dataset_name"

rules = FMARules()
rules.add(metric = "F1Score",  metric_threshold = 0.25, label = "ALL", type="label", background_label="no data", include_background=False)
rules.add(metric = "Precision",  metric_threshold = 0.22, label = "ALL", type="label", background_label="no data", include_background=True)
rules.add(metric = "Recall",  metric_threshold = 0.3, label = "ALL", type="label", background_label="no data", include_background=True)
rules.add(metric = "PixelAccuracy",  metric_threshold = 0.6, label = "ALL", type="label", background_label="no data", include_background=True)
rules.add(metric = "mIoU",  metric_threshold = 0.2, label = "ALL", type="label", background_label="no data", include_background=True)
rules.add(metric = "wIoU", metric_threshold = 0.1, weights={"water": 4, "no data": 2}, type="label", label = "ALL")

cls_default = clustering(method="k-means", embedding_col="ImageEmbedding", level="image", args= {"numOfClusters": 7})

edge_case_detection = failure_mode_analysis(test_session=test_session,
                                            dataset_name = dataset_name,
                                            test_name = "Test",
                                            model = "ModelC",
                                            gt = "GT",
                                            rules = rules,
                                            output_type="semantic_segmentation",
                                            type="embedding",
                                            clustering=cls_default
                                            )

test_session.add(edge_case_detection)

test_session.run()