import abc
import base64
import io
import os
import tempfile
import zipfile
import tarfile
import json
import argparse
from typing import List
import urllib.request
import numpy as np

from singa import singa_wrap as singa
from singa import opt
from singa import device
from singa import tensor
from singa import sonnx
from singa import layer
from singa import autograd
import onnx

from singa_auto.model import BaseModel
from singa_auto.constants import ModelDependency
from singa_auto.model.dev import make_predictions, _check_model_class, _print_header, _check_dependencies, inform_user
from singa_auto.model.utils import load_model_class
from singa_auto.advisor.constants import Proposal, ParamsType

from transformers import GPT2Tokenizer


def download_model(url):
    download_dir = '/tmp/'
    with tarfile.open(check_exist_or_download(url), 'r') as t:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(t, path=download_dir)


def check_exist_or_download(url):
    download_dir = '/tmp/'
    name = url.rsplit('/', 1)[-1]
    filename = os.path.join(download_dir, name)
    if not os.path.isfile(filename):
        urllib.request.urlretrieve(url, filename)
    return filename


class MyModel(sonnx.SONNXModel):

    def __init__(self, onnx_model):
        super(MyModel, self).__init__(onnx_model)

    def forward(self, *x):
        y = super(MyModel, self).forward(*x)
        return y[0]

    def train_one_batch(self, x, y):
        pass


class OnnxGPT2(BaseModel):

    def __init__(self, length=20, **knobs):
        super().__init__(**knobs)
        # onnx model url
        self.model_url = 'https://github.com/onnx/models/raw/master/text/machine_comprehension/gpt-2/model/gpt2-lm-head-10.tar.gz'

        # model path in the downloaded tar file
        self.model_path = 'GPT-2-LM-HEAD/model.onnx'
        self.dev = device.get_default_device()
        self.length = length
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    def train(self, dataset_path, **kwargs):
        pass

    def get_knob_config(self):
        pass

    def evaluate(self, dataset_path, **kwargs):
        pass

    def destroy(self):
        pass

    def dump_parameters(self):
        pass

    def predict(self, queries: List[str]):
        print("Get queries")

        res = []
        queries = [self._preprocess(ele) for ele in queries]

        for input_ids in queries:
            x = tensor.Tensor(device=self.dev, data=input_ids)
            out = []
            for i in range(self.length):
                y = self._model.forward(x)
                y = autograd.reshape(y, y.shape[-2:])[-1, :]
                y = tensor.softmax(y)
                y = tensor.to_numpy(y)[0]
                y = np.argsort(y)[-1]
                out.append(y)
                y = np.array([y]).reshape([1, 1, -1]).astype(np.float32)
                y = tensor.Tensor(device=self.dev, data=y)
                x = tensor.concatenate([x, y], 2)
            result = self._postprocess(out)
            res.append(result)
        return res

    def load_parameters(self, params):
        self._model = self._build_model()

    def _build_model(self):
        # read and make onnx model
        download_model(self.model_url)
        onnx_model = onnx.load(os.path.join(
            '/tmp', self.model_path))
        model = MyModel(onnx_model)
        return model

    def _preprocess(self, query: str):
        tokens = self.tokenizer.encode(query)
        tokens = np.array(tokens)
        return tokens.reshape([1, 1, -1]).astype(np.float32)

    def _postprocess(self, out: List[int]):
        text = self.tokenizer.decode(out)
        return text


if __name__ == '__main__':
    model_file_path = __file__
    model_class = 'OnnxGPT2'
    dependencies = {ModelDependency.SINGA: '3.0.0',
                    ModelDependency.ONNX: '1.15.0'}
    task = 'TEXT_GENERATION'
    queries = ['Here is some text to encode : Hello World!']

    _print_header('Installing & checking model dependencies...')
    _check_dependencies(dependencies)

    _print_header('Checking loading of model & model definition...')
    with open(model_file_path, 'rb') as f:
        model_file_bytes = f.read()
    py_model_class = load_model_class(model_file_bytes,
                                      model_class,
                                      temp_mod_name=model_class)
    _check_model_class(py_model_class)
    proposal = Proposal(trial_no=0, knobs={},
                        params_type=ParamsType.LOCAL_RECENT)

    (predictions, model_inst) = make_predictions(queries, task,
                                                 py_model_class,
                                                 proposal,
                                                 fine_tune_dataset_path=None,
                                                 params={})

    py_model_class.teardown()

    inform_user('No errors encountered while testing model!')
