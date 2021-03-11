# Speech Recognition Models

## Customizing `TfDeepSpeech`

### Using Pre-built LM and Trie

By default, the `TfDeepSpeech` model should use the pre-built language model (LM) and trie from https://github.com/mozilla/DeepSpeech/tree/v0.6.0-alpha.4/data/lm for training,
which works for the *English* language. You'll need to first download this model's file dependencies by running (in Rafiki's root folder):
    
```
bash examples/models/speech_recognition/tfdeepspeech/download_lm.sh
bash examples/models/speech_recognition/tfdeepspeech/download_trie.sh
cp examples/models/speech_recognition/tfdeepspeech/alphabet.txt tfdeepspeech/alphabet.txt
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.7.3/deepspeech-0.7.3-models.scorer -P <model_root_directory>/tfdeepspeech
```

This downloads the files `alphabet.txt`, `lm.binary`, `trie` and `scorer` into `<model_root_directory>/tfdeepspeech`, where the `TfDeepSpeech` model reads its dependencies from by default.

```
pip --no-cache-dir install tensorflow==1.12.0
pip install ds_ctcdecoder==0.6.1
```
This will download dependencies={"ds_ctcdecoder":"0.6.1", "tensorflow":'1.12.0', }

If you wish to generate your own language models and trie files instead, or wish to implement TfDeepSpeech to other languages, see instructions provided below.

### Generating Language Models

The TfDeepSpeech model requires a binary n-gram language model compiled by `kenlm` to make predictions. You can simple download pre-built lm.binary, alphabet.txt and trie files using above stpes, or should you in need to generate your own language model, please follow the steps in the example below to generate a LibriSpeech language model for English language:

1. Download the required txt.gz by running the python script

    ```sh 
    python examples/models/speech_recognition/tfdeepspeech/download_lm_txt.py
    ```

1. Install dependencies for building language model

    ```sh
    sudo apt-get install build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev
    ```

2. Build KenLM Language Model Toolkit

    ```sh
    cd /tmp/
    wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz
    mkdir kenlm/build
    cd kenlm/build
    cmake ..
    make -j2
    ```
3. Build pruned LM

    ```sh
    bin/lmplz --order 5 \
              --temp_prefix /tmp/ \
              --memory 50% \
              --text /tmp/lower.txt \
              --arpa /tmp/lm.arpa \
              --prune 0 0 0 1
    ```
    This step may take an hour to complete.
    
4. Quantize and produce trie binary

    Now substitute `<rafiki_root_directory>` with the path to rafiki root, and run the following:

    ```sh
    bin/build_binary -a 255 \
                     -q 8 \
                     trie \
                     /tmp/lm.arpa \
                     <rafiki_root_directory>/data/lm.binary
    rm /tmp/lm.arpa
    ```
    The `lm.binary` binary Language Model file is now in the data directory.
    
### Generating Trie 

See documentation on [DeepSpeech Git Repo](https://github.com/mozilla/DeepSpeech/tree/master/native_client) to generate the trie for your language model. You can simple download pre-built lm.binary, alphabet.txt and trie files using above stpes, or should you in need to generate your own language model, please follow the steps below up to **Compile libdeepspeech.so & generate_trie** section. The generated binaries will be saved to `bazel-bin/native-client/`.

Remember to modify the `alphabet.txt` file if you are training TfDeepSpeech on languages other than English.

Run

```sh
bazel-bin/native-clinet/generate_trie ../rafiki/examples/datasets/speech_recognition/alphabet.txt ../rafiki/data/lm.binary ../rafiki/data/trie
```

The `trie` file is now in the data directory.

*Note: The `generate_trie` binaries are subject to updates by the DeepSpeech team. If you find mismatch of trie file version, update the version of ctc_decoder package by amending the `VERSION` variable in `examples/models/speech_recognition/utils/taskcluster.py`.*


### Test with Sample Dataset

Run

    ```sh
    python examples/datasets/audio_files/load_sample_ldc93s1.py
    python examples/datasets/audio_files/load_librispeech.py 
    ```
to download the sample and training datasets.

### Run Test with Sample Dataset

Run the below script to install dependencies for model in the host server environment (e.g. docker container)

    ```sh
    pip install -U pip \
    && pip install -r examples/models/speech_recognition/requirements.txt
    ```

Use Python API to create model, pls run

    ```python
    import os
    from singa_auto.client import Client
    from singa_auto.constants import BudgetOption, ModelDependency
    
    # change localhost address and port number accordingly 
    # to conform with settings in web/src/HTTPconfig.js, scripts/docker_swarm/.env.sh, scripts/.base_env.sh
    client = Client(admin_host='localhost', admin_port=3000) 
    client.login(email='<USER_EMAIL>', password='<USER_PASSWORD>')

    task = 'SPEECH_RECOGNITION'
    
    # if nessacery, you can change into other dataset
    data_dir = 'data/libri'
    train_dataset_path = os.path.join(data_dir, 'dev-clean.zip') 
    
    created_model=client.create_model(name='<MODEL_NAME>',
        task='SPEECH_RECOGNITION',
        model_file_path='examples/models/speech_recognition/TfDeepSpeech.py',
        model_class='TfDeepSpeech',model_preload_file_path ='examples/models/speech_recognition/TfDeepSpeech.py',
        dependencies={"ds_ctcdecoder":"0.6.1", "tensorflow":'1.12.0', })

    budget = {BudgetOption.TIME_HOURS: 0.5, BudgetOption.GPU_COUNT: 0, BudgetOption.MODEL_TRIAL_COUNT: 1}
    
    # to create a inference job with speech_recognition model
    client.create_inference_job_by_checkpoint(model_name= created_model['name'], budget= budget)
    
    # to obtain the predictor_host
    client.get_running_inference_job(app=created_model['name'])
    
    import json
    import requests
    data = 'data/ldc93s1/ldc93s1/LDC93S1.wav'
    res = requests.post('http://{}'.format(<PREDICTOR_HOST>), json=data)
    
    # to print out the prediction result
    print(res.text)
    ```
