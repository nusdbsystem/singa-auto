from rafiki.client import Client
from pprint import pprint

# change to your own ip address and port
#client = Client(admin_host='ncrs.d2.comp.nus.edu.sg', admin_port=13000)
client = Client(admin_host='127.0.0.1', admin_port=3000)
client.login(email='superadmin@rafiki', password='rafiki')

"""
# from within docker rafiki_admin container exec -it
ret = client.create_dataset(
    name="CLIENT-traindataset",
    task="IMAGE_CLASSIFICATION",
    dataset_path="/root/OneDrive-2020-02-18/train55.zip"
)
ret = client.create_dataset(
    name="CLIENT-valdataset",
    task="IMAGE_CLASSIFICATION",
    dataset_path="/root/OneDrive-2020-02-18/val55.zip"
)

# do not change model class and dependencies
ret = client.create_model(
    name='PyPandaVgg', # MODEL_NAME    
    task='IMAGE_CLASSIFICATION',
    model_file_path='/root/examples/models/image_classification/PyPandaVgg.py',
    model_class='PyPandaVgg', # MUST BE THE MODEL_CLASS_NAME DEFINED IN .py FILE
    dependencies={ 'torch': '1.0.1', # 'tensorflow':'1.12.0', etc
                   'torchvision': '0.2.2',
                   'matplotlib': '3.1.0',
                   'lime': '0.1.1.36'}  
)
"""

# list all datasets
DATASETS = client.get_datasets()

print("\n>>>>DATASETS:")
pprint(DATASETS)

# Listing available models by task
AVAILABLE_MODELS = client.get_available_models(task='IMAGE_CLASSIFICATION')

print("\n>>>>AVAILABLE_MODELS:")
pprint(AVAILABLE_MODELS)

#"""
# call client create train job
TRAIN_JOB_RESPONSE = client.create_train_job(
    app='APP_NAME',
    task ='IMAGE_CLASSIFICATION', 
    train_dataset_id= '278927ac-e87d-485b-98d9-eb101f787b2f', #DS-Train55
    val_dataset_id= 'fc72f670-8d25-4bbe-a7e7-dff9e00e3666', #DS-Val55 
    budget ={'TIME_HOURS': 0.09,
        'GPU_COUNT': 0, 
        'MODEL_TRIAL_COUNT': 5}, 
    models=['338a6ebd-ecf2-459b-8a6c-86493321eb90'],
    train_args={}
)
#"""

# Get train_jobs
TRAIN_JOBS = client.get_train_job('APP_NAME', 1)

print("\n>>>>TRAIN_JOBS:")
pprint(TRAIN_JOBS)

# Get trials of a train job
TRIAL_TRAIN_JOB = client.get_trials_of_train_job('APP_NAME', 1)

print("\n>>>>TRIAL_TRAIN_JOB:")
pprint(TRIAL_TRAIN_JOB)

# Get trials
pprint(client.get_trial('8ff8a389-b95e-4bfb-b61d-3834a7370990'))

pprint(client.get_best_trials_of_train_job('APP_NAME'))