To create a model training job, you'll specify the train & validation datasets by their IDs, together with your application's name and its associated task.

After creating a train job, you can monitor it on SINGA-Auto Web Admin (see :ref:`using-web-admin`).

Refer to the parameters of :meth:`singa_auto.client.Client.create_train_job()` for configuring how your train job runs on SINGA-Auto, such as enabling GPU usage & specifying which models to use.

Example:

    .. code-block:: python

        client.create_train_job(
            app='fashion_mnist_app',
            task='IMAGE_CLASSIFICATION',
            train_dataset_id='ecf87d2f-6893-4e4b-8ed9-1d9454af9763',
            val_dataset_id='7e9a2f8a-c61d-4365-ae4a-601e90892b88',
            budget={ 'MODEL_TRIAL_COUNT': 5 }
            model_ids='["652db9f7-d23d-4b79-945b-a56446ceff33"]'
        )
        # Omitting the GPU_COUNT is the same as letting GPU_COUNT equal to 0, which means training will be hosted on CPU only
        # MODEL_TRIAL_COUNT stands for number of trials, minimus MODEL_TRIAL_COUNT is 1 for a valid training
        # TIME_HOURS is assigned training time limit in hours.
        # train_args={} could be left empty or unspecified, if not in use
        client.create_train_job(
            app='fashion_mnist_app',
            task='IMAGE_CLASSIFICATION',
            train_dataset_id='ecf87d2f-6893-4e4b-8ed9-1d9454af9763',
            val_dataset_id='7e9a2f8a-c61d-4365-ae4a-601e90892b88',
            budget={'TIME_HOURS': 0.01,
                    'GPU_COUNT': 0,
                    'MODEL_TRIAL_COUNT': 1}
            model_ids='["652db9f7-d23d-4b79-945b-a56446ceff33"]',
            train_args={}
        )
    Output:

    .. code-block:: python

        {'app': 'fashion_mnist_app',
        'app_version': 1,
        'id': 'ec4db479-b9b2-4289-8086-52794ffc71c8'}

Using distributed training:
    refer to https://pytorch.org/docs/stable/distributed.html

Example:

    .. code-block:: python
        # if use cpu, must set dist_model to nccl

        client.create_train_job(
            app='DistMinist',
            task='IMAGE_CLASSIFICATION',
            train_dataset_id='ecf87d2f-6893-4e4b-8ed9-1d9454af9763',
            val_dataset_id='7e9a2f8a-c61d-4365-ae4a-601e90892b88',
            budget={ 'MODEL_TRIAL_COUNT': 1, "DIST_WORKERS":3, "GPU_COUNT": 3 }
            model_ids='["652db9f7-d23d-4b79-945b-a56446ceff33"]',
            train_args={"use_dist": True,
                        "dist_model": "nccl" }
        )
        # if use cpu, must set dist_model to gloo, and dont provide GPU_COUNT
        client.create_train_job(
                              app='DistMinist2',
                              task='IMAGE_CLASSIFICATION',
                              train_dataset_id="61301c89-6e50-43f5-8662-8c195b270997",
                              val_dataset_id="61301c89-6e50-43f5-8662-8c195b270997",
                              models=['9f35bae8-5963-4ea9-b50c-31e667725937'],
                              budget={'MODEL_TRIAL_COUNT': 1,
                                      "DIST_WORKERS": 3
                                      },
                              train_args={"use_dist": True,
                                          "dist_model": "gloo"}
                              ))


    Output:

    .. code-block:: python

        {'app': 'DistMinist',
        'app_version': 1,
        'id': 'ec4db479-b9b2-4289-8086-52794ffc71c8'}
.. seealso:: :meth:`singa_auto.client.Client.create_train_job`
