from singa_auto.client import Client
import pprint

client = Client(admin_host="localhost", admin_port=3000)
client.login(email="superadmin@singaauto", password="singa_auto")


# client.create_model(
#     name='TfFeedForward',
#     task='IMAGE_CLASSIFICATION',
#     model_file_path='examples/models/image_classification/TfFeedForward.py',
#     model_class='TfFeedForward',
#     dependencies={ 'tensorflow': '1.12.0' }
# )
# #
# ret = client.get_available_models(task='IMAGE_CLASSIFICATION')
# pprint.pprint(ret)
# #
# ret = client.create_dataset(
#     name='fashion_mnist_train',
#     task='IMAGE_CLASSIFICATION',
#     dataset_path='data/fashion_mnist_train.zip'
# )
# pprint.pprint(ret)
# #
# client.create_dataset(
#     name='fashion_mnist_val',
#     task='IMAGE_CLASSIFICATION',
#     dataset_path='data/fashion_mnist_val.zip'
# )
# pprint.pprint(ret)
# #
# pprint.pprint(client.get_datasets())


# ret = client.create_train_job(
#     app='fashion_mnist_app',
#     task='IMAGE_CLASSIFICATION',
#     train_dataset_id='b7fcf14a-1e98-4614-bc78-1f4a40667937',
#     val_dataset_id='06e8ad81-2faf-4c67-9e9d-863912019eeb',
#     budget={ 'MODEL_TRIAL_COUNT': 5 }
# )
# pprint.pprint(ret)
# #
# ret = client.get_train_jobs_of_app(app='fashion_mnist_app')
# pprint.pprint(ret)
#
# ret = client.create_inference_job(app='fashion_mnist_app')
# pprint.pprint(ret)
# #
# pprint.pprint(client.get_inference_jobs_of_app(app='fashion_mnist_app'))

predictor_host = '127.0.0.1:47887'
query_path = 'examples/data/image_classification/fashion_mnist_test_1.png'

# Load query image as 3D list of pixels
from singa_auto.model import utils
[query] = utils.dataset.load_images([query_path]).tolist()

# Make request to predictor
import requests
res = requests.post('http://{}/predict'.format(predictor_host), json={ 'query': query })
print(res.json())
