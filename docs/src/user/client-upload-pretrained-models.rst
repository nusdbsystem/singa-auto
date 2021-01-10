Create a pretrain model and build inference job according to it.


Create models with pretrain id
--------------------------------------------------------------------
Example:

    .. code-block:: python

        client.create_model(
                        name='TfFeedForward',
                        task='IMAGE_CLASSIFICATION',
                        model_file_path='./examples/models/image_classification/TfFeedForward.py',
                        model_class='TfFeedForward',
                        model_preload_file_path="b42cde03-0bc3-4b15-a276-4d95f6c88fa8.model",
                        dependencies={ModelDependency.TENSORFLOW: '1.12.0'}
                    )

    Output:

    .. code-block:: python

         {'id': '82ff334a-e184-46ef-806e-709670e6713e',
         'name': 'TfFeedForward',
         'user_id': 'e21c951b-53f5-4399-95a4-796e07d9e058'}
         

Create inference based on the model
--------------------------------------------------------------------
    .. code-block:: python
        client.create_inference_job_by_checkpoint(model_name='TfFeedForward')


Test the prediction service
--------------------------------------------------------------------
    .. code-block:: python

        predictor_host = '127.0.0.1:40005'
        query_path = './examples/data/image_classification/0-3104.png'
        import requests
        files = {'img': open(query_path, 'rb')}
        res = requests.post('http://{}/predict'.format(predictor_host), files=files)
        print(res.text)



FoodLg model upload
--------------------------------------------------------------------

    .. code-block:: python

        client.create_model(
            name='Singapore Local Food - FC5Healthy-v2',
            task='IMAGE_CLASSIFICATION',
            model_file_path='./examples/models/image_object_detection/food_darknet_xception1.py',
            model_class='FoodDetection',
            model_preload_file_path="model231.zip",
            dependencies={"keras": "2.2.4", "tensorflow": "1.12.0"}
        )

Create FoodLg inference based on the model
--------------------------------------------------------------------
    .. code-block:: python

        client.create_inference_job_by_checkpoint(model_name='Singapore Local Food - FC5Healthy-v2')


QuestionAnswering model upload
--------------------------------------------------------------------
Due to the nature of 'task' of 'question_answering_covid19', corpus data 'covid19data.zip' is required for building the pre-training 'QuestionAnswering' model. 'model_preload_file_path' parameter is also used for pre-loading 'covid19data.zip' while skip the training process.
    .. code-block:: python

         client.create_model(
                           name='QAv1',
                           task='question_answering_covid19',
                           model_file_path='.../singa-easy/examples/models/question_answering/QuestionAnswering.py',
                           model_class='QuestionAnswering',
                           model_preload_file_path=".../singa-auto/data/covid19data.zip",
                           dependencies={"torch": "1.3.1", "torchvision": "0.4.2",
                           "semanticscholar": "0.1.4",
                           "sentence_transformers": "0.2.6.1",
                           "tqdm": "4.27"}
                           )

