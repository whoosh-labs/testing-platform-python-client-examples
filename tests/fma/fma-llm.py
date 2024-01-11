import datetime

from raga import clustering, FMA_LLMRules, TestSession, failure_mode_analysis, failure_mode_analysis_llm

run_name = f"run-failure-mode-llm-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", run_name= run_name, profile="dev1")


rules = FMA_LLMRules()

rules.add(metric = 'accuracy',  metric_threshold = 0.5, eval_metric='BLEU', threshold=0.5)

rules.add(metric = 'accuracy',  metric_threshold = 0.5, eval_metric='CosineSimilarity', threshold=0.5)

rules.add(metric = 'accuracy',  metric_threshold = 0.5, eval_metric='METEOR', threshold=0.4)

rules.add(metric = 'accuracy',  metric_threshold = 0.5, eval_metric='ROUGE', threshold=0.4)

dataset_name = "Enter-dataset-name"

cls_default = clustering(test_session=test_session,
                         dataset_name=dataset_name,
                         method="k-means",
                         embedding_col="summary_vector",
                         level="image",
                         args={"numOfClusters": 30}
                         )

edge_case_detection = failure_mode_analysis_llm(test_session=test_session,
                                            dataset_name=dataset_name,
                                            test_name="fma_llm_1",
                                            model="modelA",
                                            gt="GT",
                                            rules=rules,
                                            type="fma",
                                            output_type="llm",
                                            prompt_col_name = "document",
                                            model_column = "summary",
                                            gt_column = "reference_summary",
                                            embedding_col_name = "document_vector",
                                            model_embedding_column = "summary_vector",
                                            gt_embedding_column = "reference_summary_vector",
                                            clustering=cls_default
                                            )

test_session.add(edge_case_detection)

test_session.run()