from __future__ import absolute_import

import os
import sys
sys.path.append(os.getcwd())

import argparse
import base64

from pprint import pprint

from singa_auto.client import Client
# from singa_auto.config import SUPERADMIN_EMAIL
from singa_auto.constants import BudgetOption
from singa_auto.constants import InferenceBudgetOption
from singa_auto.constants import ModelDependency


from examples.scripts.quickstart import gen_id
from examples.scripts.quickstart import get_predictor_host
from examples.scripts.quickstart import make_predictions_image
from examples.scripts.quickstart import wait_until_train_job_has_stopped

SINGA_AUTO_IMAGE_NAME = f"singa_auto/singa_auto_worker"
SINGA_AUTO_VERSION = os.environ.get('SINGA_AUTO_VERSION', '0.2.0')
IMAGE_OBJECT_DETECTION_NAME = f'{SINGA_AUTO_IMAGE_NAME}:{SINGA_AUTO_VERSION}'


def run_object_detection(client, train_dataset_path, val_dataset_path, gpus, hours, **kwargs):
    '''
    Conducts training with the `YoloV3` model for the task ``OBJECT_DETECTION`.
    '''

    task = 'OBJECT_DETECTION'

    import time

    if "train_dataset" in kwargs:
        train_dataset = kwargs["train_dataset"]
    else:
        print('Creating & uploading train dataset onto SINGA-Auto...')
        curr_time = time.strftime("%m%d_%H%M", time.localtime(time.time()))
        train_dataset = client.create_dataset('yolo_train_{}'.format(curr_time), task, train_dataset_path)
    pprint(train_dataset)

    if "val_dataset" in kwargs:
        val_dataset = kwargs["val_dataset"]
    else:
        print('Creating & uploading val dataset onto SINGA-Auto...')
        curr_time = time.strftime("%m%d_%H%M", time.localtime(time.time()))
        val_dataset = client.create_dataset('yolo_val_{}'.format(curr_time), task, val_dataset_path)
    pprint(val_dataset)

    if "train_model" in kwargs:
        train_model = kwargs["train_model"]
    else:
        curr_time = time.strftime("%m%d_%H%M", time.localtime(time.time()))
        model_name = 'yolo_{}_iter_2'.format(curr_time)
        print('Adding models "{}" to SINGA-Auto...'.format(model_name))
        train_model = client.create_model(
            model_name,
            task,
            'examples/models/image_object_detection/SaYolo.py',
            'SaYolo',
            docker_image=IMAGE_OBJECT_DETECTION_NAME,
            dependencies={
                "opencv-python":"4.4.0.46",
                "terminaltables":"3.1.0",
                "torch":"1.6.0",
                "torchvision":"0.7.0",
                "tqdm":"4.53.0",
                "wget":"3.2",
                "pycocotools":"2.0.2",
            })
    pprint(train_model)

    # generate app & model names by time to avoid naming conflicts
    curr_time = time.strftime("%m%d_%H%M", time.localtime(time.time()))
    app = 'yolo_{}_gpu_{}'.format(curr_time, gpus)

    print('Creating train job for app "{}" on SINGA-Auto...'.format(app))
    budget = {BudgetOption.TIME_HOURS: hours, BudgetOption.GPU_COUNT: gpus}
    train_job = client.create_train_job(
        app,
        task,
        train_dataset['id'],
        val_dataset['id'],
        budget,
        models=[train_model['id']]
    )
    pprint(train_job)

    print('Waiting for train job to complete...')
    print('This might take a few minutes')
    wait_until_train_job_has_stopped(client, app)
    print('Train job has been stopped')

    # app = "yolo_0601_0935_gpu_1"

    print('Listing best trials of latest train job for app "{}"...'.format(app))
    pprint(client.get_best_trials_of_train_job(app))

    print('Creating inference job for app "{}" on SINGA-Auto...'.format(app))
    budget = {InferenceBudgetOption.GPU_COUNT: gpus}
    pprint(client.create_inference_job(app, budget=budget))
    predictor_host = get_predictor_host(client, app)
    if not predictor_host:
        raise Exception('Inference job has errored or stopped')
    print('Inference job is running!')

    print('Making predictions for queries:')
    queries = ['./examples/data/object_detection/cat.jpg']
    print(queries)
    predictions = make_predictions_image(client, predictor_host, queries)
    print('Predictions are:')
    print(predictions)

    print('Stopping inference job...')
    pprint(client.stop_inference_job(app))


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--email',
        type=str,
        default="superadmin@singaauto",
        help='Email of user',
    )
    parser.add_argument(
        '--password',
        type=str,
        default="singa_auto",
        help='Password of user',
    )
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='How many GPUs to use',
    )
    parser.add_argument(
        '--hours',
        type=float,
        default=1,
        help='How long the train job should run for (in hours)',
    )
    parser.add_argument(
        '--use_old',
        type=bool,
        default=True,
        help='whether use existing dataset and model',
    )
    (args, _) = parser.parse_known_args()

    # Initialize client
    client = Client()
    client.login(email=args.email, password=args.password)

    print('Preprocessing dataset...')
    data_dir = '/home/taomingyang/dataset/package'
    train_dataset_path = os.path.join(data_dir, 'coco_cat.zip')
    val_dataset_path = os.path.join(data_dir, 'coco_mini.zip')

    if args.use_old:
        train_dataset = {
            'id': 'a5181e1f-74d1-4916-a853-4ab75afa81d5',
            'name': 'yolo_train_0531_1650',
            'owner_id': '6d37f19f-9063-4b47-a73f-5cb6577f4f85',
            'size_bytes': 573851732,
            'stat': {},
            'store_dataset_id': '039e05a7-917a-4441-aaff-110cf7552c73.data',
            'task': 'OBJECT_DETECTION'
        }
        val_dataset = {
            'id': '3bb9113f-2339-4cd4-9c22-d99ffcdd27b9',
            'name': 'yolo_val_0531_1650',
            'owner_id': '6d37f19f-9063-4b47-a73f-5cb6577f4f85',
            'size_bytes': 24435329,
            'stat': {},
            'store_dataset_id': '93f4c5d2-b9d8-4585-b193-c1d19bb8026d.data',
            'task': 'OBJECT_DETECTION'
        }
        # train_model = { # using server dataset
        #     'id': '48cd0413-ec4b-4f9b-8364-bbb51db52a45',
        #     'name': 'yolo_0512_1055_iter_1',
        #     'user_id': 'dd703056-e2f2-4e30-9e44-ecd1f7ccee7d'
        # }
        train_model = { # using local dataset, mini train
            'id': 'dcd9e8a0-e74f-4ca1-880c-864405a91f92',
            'name': 'yolo_0531_1650_iter_2',
            'user_id': '6d37f19f-9063-4b47-a73f-5cb6577f4f85'
        }
        # train_model = { # using local dataset, cat train
        #     'id': '4ac4b8bc-7225-46bd-8f50-786434cf0d3e',
        #     'name': 'yolo_0520_0855_iter_2',
        #     'user_id': 'dd703056-e2f2-4e30-9e44-ecd1f7ccee7d'
        # }

        run_object_detection(
            client, train_dataset_path, val_dataset_path, args.gpus, args.hours,
            train_dataset=train_dataset, val_dataset=val_dataset, train_model=train_model,
        )
    else:
        run_object_detection(client, train_dataset_path, val_dataset_path, args.gpus, args.hours)

    print(args)
