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

from examples.models.question_answering.onnx_bert import tokenization
from examples.models.question_answering.onnx_bert.run_onnx_squad import parse_predictions, convert_examples_to_features, RawResult, parse_squad_examples


def download_model(url):
    download_dir = '/tmp/'
    with tarfile.open(check_exist_or_download(url), 'r') as t:
        t.extractall(path=download_dir)


def check_exist_or_download(url):
    download_dir = '/tmp/'
    name = url.rsplit('/', 1)[-1]
    filename = os.path.join(download_dir, name)
    if not os.path.isfile(filename):
        urllib.request.urlretrieve(url, filename)
    return filename


def load_vocab():
    url = 'https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip'
    download_dir = '/tmp/'
    filename = os.path.join(
        download_dir, 'uncased_L-12_H-768_A-12', 'vocab.txt')
    with zipfile.ZipFile(check_exist_or_download(url), 'r') as z:
        z.extractall(path=download_dir)
    return filename


class MyModel(sonnx.SONNXModel):

    def __init__(self, onnx_model):
        super(MyModel, self).__init__(onnx_model)

    def forward(self, *x):
        y = super(MyModel, self).forward(*x)
        return y

    def train_one_batch(self, x, y):
        pass


class OnnxBert(BaseModel):

    def __init__(self, device_id=0, length=20, **knobs):
        super().__init__(**knobs)
        # onnx model url
        self.model_url = 'https://media.githubusercontent.com/media/onnx/models/master/text/machine_comprehension/bert-squad/model/bertsquad-10.tar.gz'

        # model path in the downloaded tar file
        self.model_path = 'download_sample_10/bertsquad10.onnx'
        self.dev = device.create_cuda_gpu_on(device_id)
        self.max_answer_length = 30
        self.max_seq_length = 256
        self.doc_stride = 128
        self.max_query_length = 64
        self.n_best_size = 20
        self.batch_size = 3

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

    def predict(self, queries: json):
        print("Get queries")

        bs = self.batch_size

        res = []
        input_ids, input_mask, segment_ids, extra_data, eval_examples = self._preprocess(
            queries)
        n = len(input_ids) // bs
        all_results = []

        tmp_dict = {}
        for idx in range(0, n):
            inputs = [
                np.array([eval_examples[idx].qas_id for idx in range(
                    idx, idx+bs)], dtype=np.int32),
                segment_ids[idx:idx + bs].astype(np.int32),
                input_mask[idx:idx + bs].astype(np.int32),
                input_ids[idx:idx + bs].astype(np.int32),
            ]

            x_batch = []
            for inp in inputs:
                tmp_tensor = tensor.from_numpy(inp)
                tmp_tensor.to_device(self.dev)
                x_batch.append(tmp_tensor)

            outputs = self._model.forward(*x_batch)

            result = []
            for outp in outputs:
                result.append(tensor.to_numpy(outp))

            in_batch = result[1].shape[0]
            start_logits = [float(x) for x in result[1][0].flat]
            end_logits = [float(x) for x in result[0][0].flat]
            for i in range(0, in_batch):
                unique_id = len(all_results)
                all_results.append(
                    RawResult(unique_id=unique_id,
                              start_logits=start_logits,
                              end_logits=end_logits))

        return self._postprocess(eval_examples, extra_data, all_results)

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
        vocab_file = load_vocab()
        tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file,
                                               do_lower_case=True)

        eval_examples = parse_squad_examples(query)

        # Use convert_examples_to_features method from run_onnx_squad to get parameters from the input
        input_ids, input_mask, segment_ids, extra_data = convert_examples_to_features(
            eval_examples, tokenizer, self.max_seq_length, self.doc_stride, self.max_query_length)
        return input_ids, input_mask, segment_ids, extra_data, eval_examples

    def _postprocess(self, eval_examples, extra_data, all_results):
        return parse_predictions(eval_examples, extra_data, all_results, self.n_best_size,
                                 self.max_answer_length, True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--query_path',
        type=str,
        default='examples/data/question_answering/inputs.json',
        help='Path(s) to a query question json')
    (args, _) = parser.parse_known_args()
    input_file = args.query_path
    model_file_path = __file__
    model_class = 'OnnxBert'
    dependencies = {ModelDependency.SINGA: '3.0.0',
                    ModelDependency.ONNX: '1.15.0'}
    task = 'QUESTION_ANSWERING'
    with open(input_file, "r") as f:
        queries = json.load(f)["data"]

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
                                                 proposal, params={})

    py_model_class.teardown()

    inform_user('No errors encountered while testing model!')
