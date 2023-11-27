from raga import *
import datetime

run_name = f"Nano_Net_Test-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", run_name= run_name, profile="raga-dev-new")

dataset_name = "Enter_your_dataset_name"

filters = Filter()
filters.add(TimestampFilter(gte="2021-07-07T00:00:00Z", lte="2024-07-08T00:00:00Z"))


rules = EventABTestRules()
rules.add(metric = "difference_percentage", IoU = 0.5, _class = "ALL", threshold = 0.55, conf_threshold=0.5)
rules.add(metric = "difference_count", IoU = 0.5, _class = "ALL", threshold = 1.0, conf_threshold=0.5)

model_comparison_check = event_ab_test(test_session=test_session,
                                       dataset_name=dataset_name,
                                       test_name="AB-test-unlabelled",
                                       modelB = "Complex-America-Stop-0.0.1",
                                       modelA = "Production-America-Stop-0.0.1",
                                       object_detection_modelB="Complex-America-Stop-0.0.1",
                                       object_detection_modelA = "Production-America-Stop-0.0.1",
                                       type = "metadata",
                                       sub_type = "unlabelled",
                                       output_type = "event_detection",
                                       rules = rules,
                                       aggregation_level=["weather","scene","time_of_day"],
                                       filter=filters)

test_session.add(model_comparison_check)

test_session.run()