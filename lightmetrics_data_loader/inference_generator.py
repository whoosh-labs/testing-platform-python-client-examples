from raga import *
import datetime


run_name = f"lm-inference-test-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"


# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", run_name= run_name, profile="lm-dev")


filters = Filter()
filters.add(TimestampFilter(gte="2020-03-15T00:00:00Z", lte="2050-03-15T00:00:00Z"))

# #Model: Production-Vienna-Alto
# #Version: 0.0.1
# lightmetrics_inference_generator(test_session=test_session, 
#                    dataset_name="test-lm-loader-3-nov-v2", 
#                    filter=filters, 
#                    model_name="Production-Vienna-Alto-0.0.1", 
#                    event_inference_col_name="Production-Vienna-Alto-Event", 
#                    model_inference_col_name="Production-Vienna-Alto-Model")

# #Model: Production-Canada-Stop-Quebec
# #Version: 0.0.1
# lightmetrics_inference_generator(test_session=test_session, 
#                    dataset_name="test-lm-loader-3-nov-v2", 
#                    filter=filters, 
#                    model_name="Production-Canada-Stop-Quebec-0.0.1", 
#                    event_inference_col_name="Production-Canada-Stop-Quebec-Event", 
#                    model_inference_col_name="Production-Canada-Stop-Quebec-Model")

#Model: Production-America-Stop
#Version: 0.0.1
lightmetrics_inference_generator(test_session=test_session, 
                   dataset_name="test-lm-loader-6-nov-v4", 
                   filter=filters, 
                   model_name="Production-America-Stop-0.0.1", 
                   event_inference_col_name="Production-America-Stop-Event", 
                   model_inference_col_name="Production-America-Stop-Model")

# #Model: Production-Vienna-Stop
# #Version: 0.0.1
# lightmetrics_inference_generator(test_session=test_session, 
#                    dataset_name="test-lm-loader-3-nov-v2", 
#                    filter=filters, 
#                    model_name="Production-Vienna-Stop-0.0.1", 
#                    event_inference_col_name="Production-Vienna-Stop-Event", 
#                    model_inference_col_name="Production-Vienna-Stop-Model")


# #Model: Complex-Vienna-Alto
# #Version: 0.0.1
# lightmetrics_inference_generator(test_session=test_session, 
#                    dataset_name="test-lm-loader-3-nov-v2", 
#                    filter=filters, 
#                    model_name="Complex-Vienna-Alto-0.0.1", 
#                    event_inference_col_name="Complex-Vienna-Alto-Event", 
#                    model_inference_col_name="Complex-Vienna-Alto-Model")

# #Model: Complex-Canada-Stop-Quebec
# #Version: 0.0.1
# lightmetrics_inference_generator(test_session=test_session, 
#                    dataset_name="test-lm-loader-3-nov-v2", 
#                    filter=filters, 
#                    model_name="Complex-Canada-Stop-Quebec-0.0.1", 
#                    event_inference_col_name="Complex-Canada-Stop-Quebec-Event", 
#                    model_inference_col_name="Complex-Canada-Stop-Quebec-Model")

# #Model: Complex-America-Stop
# #Version: 0.0.1
# lightmetrics_inference_generator(test_session=test_session, 
#                    dataset_name="test-lm-loader-3-nov-v2", 
#                    filter=filters, 
#                    model_name="Complex-America-Stop-0.0.1", 
#                    event_inference_col_name="Complex-America-Stop-Event", 
#                    model_inference_col_name="Complex-America-Stop-Model")

# #Model: Complex-Vienna-Stop
# #Version: 0.0.1
# lightmetrics_inference_generator(test_session=test_session, 
#                    dataset_name="test-lm-loader-3-nov-v2", 
#                    filter=filters, 
#                    model_name="Complex-Vienna-Stop-0.0.1", 
#                    event_inference_col_name="Complex-Vienna-Stop-Event", 
#                    model_inference_col_name="Complex-Vienna-Stop-Model")