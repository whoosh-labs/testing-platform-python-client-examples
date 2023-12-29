from raga import *


test_ds_image_url = "https://ragatesitng-dev-storage.s3.ap-south-1.amazonaws.com/coco_test_jg/test"
train_ds_image_url = "https://ragatesitng-dev-storage.s3.ap-south-1.amazonaws.com/coco_test_jg/train"


def parse_json(json_file, output_file, image_url):
    with open(json_file, 'r') as f:
        annotations = json.load(f)
        
    for image in annotations["images"]:
        image["coco_url"] = f"{image_url}/{image['file_name']}"
        
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)

parse_json("./assets/_annotations_train.coco.json", "./assets/_annotations_train_min.coco.json", train_ds_image_url)
parse_json("./assets/_annotations_test.coco.json", "./assets/_annotations_test_min.coco.json", test_ds_image_url)


test_session = TestSession(project_name="testingProject", profile="dev")

#create test_ds object of Dataset instance
test_ds = Dataset(test_session=test_session, 
                  name="coco_train_ds_v2", 
                  type=DATASET_TYPE.IMAGE,
                  format="coco",
                  data="./assets/_annotations_train_min.coco.json", 
                  model_name="ModelA",
                  inference_col_name="annotations")

test_ds = Dataset(test_session=test_session, 
                  name="coco_test_ds_v2", 
                  type=DATASET_TYPE.IMAGE,
                  format="coco",
                  data="./assets/_annotations_test_min.coco.json", 
                  model_name="ModelA",
                  inference_col_name="annotations")


test_ds.load()










# #load schema and pandas data frame
# test_ds.load(data="macoco.json", format="coco", model_name="modelA", inference_col_name="annotations1")
# test_ds.load(data="mbcoco.json", format="coco", model_name="modelB", inference_col_name="annotations2")
# test_ds.load(data="gtcoco.json", format="coco", model_name="modelG", inference_col_name="annotations3")



# # Params for labelled AB Model Testing
# testName = StringElement("TestingP-unlabelled-test-1")
# modelA = StringElement("modelA")
# modelB = StringElement("modelB")
# gt = StringElement("modelG")
# type = ModelABTestTypeElement("labelled")
# rules = ModelABTestRules()
# rules.add(metric = StringElement("precision_diff_all"), IoU = FloatElement(0.5), _class = StringElement("ALL"), threshold = FloatElement(0.5))
# rules.add(metric = StringElement("precision_candy"), IoU = FloatElement(0.5), _class = StringElement("candy"), threshold = FloatElement(0.5))
# #create payload for model ab testing
# model_comparison_check = model_ab_test(test_session, dataset_name=dataset_name, testName=testName, modelA = modelA , modelB = modelB , type = type, rules = rules, gt=gt)

# #add payload into test_session object
# test_session.add(model_comparison_check)

# #run added ab test model payload
# # test_session.run()

# # Params for unlabelled AB Model Testing
# testName = StringElement("25July_TestCOCO_KB_01")
# modelA = StringElement("modelA")
# modelB = StringElement("modelB")
# type = ModelABTestTypeElement("unlabelled")
# rules = ModelABTestRules()
# rules.add(metric = StringElement("difference_all"), IoU = FloatElement(0.5), _class = StringElement("ALL"), threshold = FloatElement(0.5))
# rules.add(metric = StringElement("difference_candy"), IoU = FloatElement(0.5), _class = StringElement("candy"), threshold = FloatElement(0.5))
# rules.add(metric = StringElement("difference_drink"), IoU = FloatElement(0.5), _class = StringElement("drink"), threshold = FloatElement(0.5))

# #create payload for model ab testing
# model_comparison_check = model_ab_test(test_session, dataset_name=dataset_name, testName=testName, modelA = modelA , modelB = modelB , type = type, rules = rules)

# #add payload into test_session object
# test_session.add(model_comparison_check)

# #run added ab test model payload
# test_session.run()