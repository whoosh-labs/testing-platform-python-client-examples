from raga import *
test_session = TestSession(project_name="testingProject", run_name= "LM-Model-Upload-Run-Test-v3", access_key="sO8bDh4uNmduIBZ4j8Mv", secret_key="CNBaK1XdT72HIrt63oEIQ62yA4ZK2Vm1pdLYNEMH", host="http://43.204.101.13:8080")

lightmetrics_model_upload(test_session=test_session, file_path="./Complex-America-Stop.zip", name="Complex-America-Stop", version="0.0.2.delete")