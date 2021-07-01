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
IMAGE_SEGMENTATION_NAME = f'{SINGA_AUTO_IMAGE_NAME}:{SINGA_AUTO_VERSION}'


def run_image_segmentation(client, dataset_path, gpus, hours, **kwargs):
    '''
    Conducts training with the `YoloV3` model for the task ``OBJECT_DETECTION`.
    '''

    task = 'IMAGE_SEGMENTATION'

    import time

    if "dataset" in kwargs:
        dataset = kwargs["dataset"]
    else:
        print('Creating & uploading train dataset onto SINGA-Auto...')
        curr_time = time.strftime("%m%d_%H%M", time.localtime(time.time()))
        dataset = client.create_dataset('oxford_pets_{}'.format(curr_time), task, dataset_path)
    pprint(dataset)

    if "model" in kwargs:
        model = kwargs["model"]
    else:
        curr_time = time.strftime("%m%d_%H%M", time.localtime(time.time()))
        model_name = 'deeplab_{}_iter_{}'.format(curr_time, 1)
        print('Adding models "{}" to SINGA-Auto...'.format(model_name))
        model = client.create_model(
            model_name,
            task,
            'examples/models/image_segmentation/SaDeeplab.py',
            'SaDeeplab',
            docker_image=IMAGE_SEGMENTATION_NAME,
            dependencies={
                "opencv-python":"4.4.0.46",
                "tensorflow": "2.3.0",
            }
        )
    pprint(model)

    # generate app & model names by time to avoid naming conflicts
    curr_time = time.strftime("%m%d_%H%M", time.localtime(time.time()))
    app = 'deeplab_{}_gpu_{}'.format(curr_time, gpus)

    print('Creating train job for app "{}" on SINGA-Auto...'.format(app))
    budget = {BudgetOption.TIME_HOURS: hours, BudgetOption.GPU_COUNT: gpus}
    train_job = client.create_train_job(
        app,
        task,
        dataset['id'],
        dataset['id'],
        budget,
        models=[model['id']]
    )
    pprint(train_job)

    print('Waiting for train job to complete...')
    print('This might take a few minutes')
    wait_until_train_job_has_stopped(client, app)
    print('Train job has been stopped')

    # app = "deeplab_0519_1026_gpu_4"

    print('Listing best trials of latest train job for app "{}"...'.format(app))
    pprint(client.get_best_trials_of_train_job(app))

    print('Creating inference job for app "{}" on SINGA-Auto...'.format(app))
    budget = {InferenceBudgetOption.GPU_COUNT: 1}
    pprint(client.create_inference_job(app, budget=budget))
    predictor_host = get_predictor_host(client, app)
    if not predictor_host:
        raise Exception('Inference job has errored or stopped')
    print('Inference job is running!')

    print('Making predictions for queries:')
    queries = ['./examples/data/image_segmentaion/Persian_120.jpg']
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
    dataset_path = os.path.join(data_dir, 'oxford_pets.zip')

    if args.use_old:
        dataset = {
            'id': '0e6723fa-7e3e-4942-9808-07b9873b2244',
            'name': 'oxford_pets_0518_1643',
            'owner_id': 'cabd4ec6-3911-4439-b88b-660eaa7d7ad8',
            'size_bytes': 401767917,
            'stat': {},
            'store_dataset_id': '4edfa4cc-5d5e-431b-a893-0bcebf653fd0.data',
            'task': 'IMAGE_SEGMENTATION'
        }
        model = {
            'id': '0a3a6bc9-a3ab-4ec7-8b4c-585af2fec948',
            'name': 'deeplab_0519_1331_iter_1',
            'user_id': 'cabd4ec6-3911-4439-b88b-660eaa7d7ad8'
        }
        # model = {
        #     'id': '6302dbe8-22c2-4b39-bd09-dd29ffed254d',
        #     'name': 'yolo_0427_1404_iter_10',
        #     'user_id': '8e29b96b-ea16-4595-a1fd-86decddbab6b'
        # }

        run_image_segmentation(
            client, dataset_path, args.gpus, args.hours,
            dataset=dataset, model=model,
        )
    else:
        run_image_segmentation(client, dataset_path, args.gpus, args.hours)

    print(args)
