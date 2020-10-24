#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

from pprint import pprint
import argparse
import os
import base64

from singa_auto.client import Client
from singa_auto.config import SUPERADMIN_EMAIL
from singa_auto.constants import BudgetOption, ModelDependency

from examples.scripts.quickstart import get_predictor_host, \
    wait_until_train_job_has_stopped, make_predictions, gen_id

from examples.datasets.audio_files.load_librispeech import load_librispeech

IMAGE_TFDEEPSPEECH_VERSION = os.environ['SINGA_AUTO_VERSION']
IMAGE_TFDEEPSPEECH = f'singa_auto_tfdeepspeech:{SINGA_AUTO_VERSION}'


def run_speech_recognition(client, train_dataset_path, val_dataset_path, gpus,
                           hours):
    '''
        Conducts training with the `TfDeepSpeech` model for the task ``SPEECH_RECOGNITION`.
    '''

    task = 'SPEECH_RECOGNITION'

    # Randomly generate app & model names to avoid naming conflicts
    app_id = gen_id()
    app = 'speech_recognition_app_{}'.format(app_id)
    tf_model_name = 'TfDeepSpeech_{}'.format(app_id)

    print('Creating & uploading datasets onto SINGA-Autp...')
    train_dataset = client.create_dataset('{}_train'.format(app), task,
                                          train_dataset_path)
    pprint(train_dataset)
    val_dataset = client.create_dataset('{}_val'.format(app), task,
                                        val_dataset_path)
    pprint(val_dataset)

    print('Adding models "{}" to SINGA-Auto...'.format(tf_model_name))
    tf_model = client.create_model(
        tf_model_name,
        task,
        'examples/models/speech_recognition/TfDeepSpeech.py',
        'TfDeepSpeech',
        docker_image=IMAGE_TFDEEPSPEECH,
        dependencies={
            ModelDependency.TENSORFLOW: '1.12.0',
            ModelDependency.DS_CTCDECODER: '0.6.0-alpha.4'
        })
    pprint(tf_model)

    print('Creating train job for app "{}" on SINGA-Autp...'.format(app))
    budget = {BudgetOption.TIME_HOURS: hours, BudgetOption.GPU_COUNT: gpus}
    train_job = client.create_train_job(app,
                                        task,
                                        train_dataset['id'],
                                        val_dataset['id'],
                                        budget,
                                        models=[tf_model['id']])
    pprint(train_job)

    print('Monitor the train job on SINGA-Autp Web Admin')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--email',
                        type=str,
                        default=SUPERADMIN_EMAIL,
                        help='Email of user')
    parser.add_argument('--password',
                        type=str,
                        default=os.environ.get('SUPERADMIN_PASSWORD'),
                        help='Password of user')
    parser.add_argument('--gpus',
                        type=int,
                        default=1,
                        help='How many GPUs to use')
    parser.add_argument('--hours',
                        type=float,
                        default=12,
                        help='How long the train job should run for (in hours)')
    (args, _) = parser.parse_known_args()

    # Initialize client
    client = Client()
    client.login(email=args.email, password=args.password)

    print('Preprocessing dataset...')
    data_dir = 'data/libri'
    parts_to_load = ['train-clean-100', 'dev-clean']
    load_librispeech(data_dir, parts=parts_to_load)
    train_dataset_path = os.path.join(data_dir, 'train-clean-100.zip')
    val_dataset_path = os.path.join(data_dir, 'dev-clean.zip')

    run_speech_recognition(client, train_dataset_path, val_dataset_path,
                           args.gpus, args.hours)
