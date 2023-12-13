from raga import *

# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", run_name= "LM-Model-Upload", profile="lm-prod")

# lightmetrics_model_upload(test_session=test_session, file_path="/home/ubuntu/developments/testing-platform-python-client/raga/examples/assets/lm_models/Complex-America-Stop.zip", name="Complex-America-Stop", version="0.0.1")

# lightmetrics_model_upload(test_session=test_session, file_path="/home/ubuntu/developments/testing-platform-python-client/raga/examples/assets/lm_models/Complex-Canada-Stop-Quebec.zip", name="Complex-Canada-Stop-Quebec", version="0.0.1")

# lightmetrics_model_upload(test_session=test_session, file_path="/home/ubuntu/developments/testing-platform-python-client/raga/examples/assets/lm_models/Complex-Vienna-Alto.zip", name="Complex-Vienna-Alto", version="0.0.1")

# lightmetrics_model_upload(test_session=test_session, file_path="/home/ubuntu/developments/testing-platform-python-client/raga/examples/assets/lm_models/Complex-Vienna-Stop.zip", name="Complex-Vienna-Stop", version="0.0.1")


# lightmetrics_model_upload(test_session=test_session, file_path="/home/ubuntu/developments/testing-platform-python-client/raga/examples/assets/lm_models/Production-America-Stop.zip", name="Production-America-Stop", version="0.0.1")

# lightmetrics_model_upload(test_session=test_session, file_path="/home/ubuntu/developments/testing-platform-python-client/raga/examples/assets/lm_models/Production-Canada-Stop-Quebec.zip", name="Production-Canada-Stop-Quebec", version="0.0.1")

lightmetrics_model_upload(test_session=test_session, file_path="/home/ubuntu/developments/testing-platform-python-client/raga/examples/assets/lm_models/Production-Vienna-Alto.zip", name="Production-Vienna-Alto", version="0.0.1")

# lightmetrics_model_upload(test_session=test_session, file_path="/home/ubuntu/developments/testing-platform-python-client/raga/examples/assets/lm_models/Production-Vienna-Stop.zip", name="Production-Vienna-Stop", version="0.0.1")


