import datetime

from raga import clustering, TestSession, FMARules, failure_mode_analysis

run_name = f"run-failure-mode-fma-instance-segmentation-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", run_name=run_name, profile="dev1")


rules = FMARules()

rules.add(metric='F1Score',  metric_threshold=0.5, label='ALL', conf_threshold=0.5, iou_threshold=0.5)

rules.add(metric='Recall',  metric_threshold=0.5,  label='ALL', conf_threshold=0.5, iou_threshold=0.5)

rules.add(metric='Precision',  metric_threshold=0.5, label='ALL', conf_threshold=0.5, iou_threshold=0.5)

dataset_name = "fma-instance-emb_6feb-v1"

cls_default = clustering(test_session=test_session,
                         dataset_name=dataset_name,
                         method="k-means",
                         embedding_col="embedding",
                         level="image",
                         args={"numOfClusters": 30}
                         )

edge_case_detection = failure_mode_analysis(test_session=test_session,
                                                dataset_name=dataset_name,
                                                test_name="fma-instance-segmentation",
                                                model="yolov8",
                                                gt="GT",
                                                rules=rules,
                                                output_type="instance_segmentation",
                                                type="fma",
                                                clustering=cls_default,
                                                embedding_col_name="embedding",
                                            )
test_session.add(edge_case_detection)

test_session.run()