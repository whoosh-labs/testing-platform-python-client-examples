from raga import *
import datetime
from raga._tests import clustering

run_name = f"FMA-SD-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
dataset_name = "fma_sd_dataset_test_embed_v25"

test_session = TestSession(project_name="testingProject", run_name= run_name, profile="dev1")

rules = SDRules()
rules.add(metric="Accuracy", label=["All"], metric_threshold=0.95)

cls_default = clustering(test_session=test_session,
                         dataset_name=dataset_name,
                         method="k-means",
                         embedding_col="embedding",
                         level="image",
                         args={"numOfClusters": 9}
                         )


edge_case_detection = fma_structured_data(test_session=test_session,
                                            dataset_name = "fma_sd_dataset_test_embed_v25",
                                            test_name = "FMA SD",
                                            type = "fma",
                                            output_type="structured_data",
                                            embedding= "embedding",
                                            model = "model_pred",
                                            gt = "repay_fail",
                                            rules = rules,
                                            clustering = cls_default
                                             )
test_session.add(edge_case_detection)

test_session.run()