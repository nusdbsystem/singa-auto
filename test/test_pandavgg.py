from rafiki.client import Client
import pprint


client = Client(admin_host="localhost", admin_port=3000)
client.login(email="superadmin@rafiki", password="rafiki")

# ret = client.create_model(
#     name='PyPandaVgg',
#     task='IMAGE_CLASSIFICATION',
#     model_file_path='examples/models/image_classification/PyPandaVgg.py',
#     model_class='PyPandaVgg',
#     dependencies={ 'torch': '1.0.1',
#                    'torchvision': '0.2.2',
#                    'matplotlib': '3.1.0',
#                    'lime': '0.1.1.36'}
# )
# # 
# ret = client.get_available_models(task='IMAGE_CLASSIFICATION')
# pprint.pprint(ret)
# # 
# ret = client.create_dataset(
#     name='PandaTrain',
#     task='IMAGE_CLASSIFICATION',
#     dataset_path='data/train.zip'
# )
# pprint.pprint(ret)
# train_id = ret['id']
# # 
# client.create_dataset(
#     name='PantaValid',
#     task='IMAGE_CLASSIFICATION',
#     dataset_path='data/valid.zip'
# )
# pprint.pprint(ret)
# valid_id = ret['id']
# # 
# pprint.pprint(client.get_datasets())
# # 
# # 
# ret = client.create_train_job(
#     app='PandaApp',
#     task='IMAGE_CLASSIFICATION',
#     train_dataset_id=train_id,
#     val_dataset_id=valid_id,
#     budget={ 'MODEL_TRIAL_COUNT': 1 }
# )
# pprint.pprint(ret)
# # 
# ret = client.get_train_jobs_of_app(app='PandaApp')
# pprint.pprint(ret)
# # 
pprint.pprint(client.stop_inference_job(app="PandaApp"))
ret = client.create_inference_job(app='PandaApp')
predictor_host = ret['predictor_host']
print(predictor_host)
# 
#predictor_host = '127.0.0.1:56319'
query_path = 'data/IM-0001-0001.jpeg'
# query_path = 'data/7d8k.jpeg'
# 
# Load query image as 3D list of pixels
from rafiki.model import utils
[query] = utils.dataset.load_images([query_path]).tolist()
# 
# Make request to predictor
import requests
res = requests.post('http://{}/predict'.format(predictor_host), json={ 'query': query })
print(res.json())