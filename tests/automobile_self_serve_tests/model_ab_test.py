from raga import *
import datetime

run_name = f"model_ab_test-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

# create test_session object of TestSession instance
test_session = TestSession(project_name="testingProject", run_name= run_name, profile="raga-dev-new")


# Params for labelled AB Model Testing
testName = "QA_Labelled_19SepV1"
modelA = "modelA"
modelB = "modelB"
gt = "modelGT"
type = "labelled"
aggregation_level = ["Reflection","Overlap","CameraAngle"]
rules = ModelABTestRules()
rules.add(metric ="precision", IoU = 0.5, _class = "ALL", threshold = 0.5)
rules.add(metric ="f1score", IoU = 0.5, _class = "ALL", threshold = 0.5)
rules.add(metric ="recall", IoU = 0.5, _class = "ALL", threshold = 0.5)
# rules.add(metric = "precision", IoU = 0.5, _class = "candy", threshold = 0.5)

#create payload for model ab testing
model_comparison_check = model_ab_test(test_session, dataset_name=dataset_name, test_name=testName, modelA = modelA , modelB = modelB ,aggregation_level = aggregation_level, type = type,  rules = rules, gt=gt)


#add payload into test_session object
test_session.add(model_comparison_check)


# # Params for unlabelled AB Model Testing
testName = "QA_Unlabelled_19SepV1"
modelA = "modelA"
modelB = "modelB"
type = "unlabelled"
aggregation_level = ["Reflection","Overlap","CameraAngle"]
rules = ModelABTestRules()
rules.add(metric ="difference_all", IoU = 0.5, _class = "ALL", threshold = 0.5)
rules.add(metric = "difference_candy", IoU = 0.5, _class = "candy", threshold = 0.5)


# #create payload for model ab testing
model_comparison_check = model_ab_test(test_session, dataset_name="automobile-dataset", test_name=testName, modelA = modelA , modelB = modelB , type = type, aggregation_level = aggregation_level, rules = rules)


# #add payload into test_session object
test_session.add(model_comparison_check)

#run added ab test model payload
test_session.run()