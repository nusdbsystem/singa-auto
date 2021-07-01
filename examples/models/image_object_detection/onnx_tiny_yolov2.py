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
import PIL
from PIL import Image

from singa import singa_wrap as singa
from singa import opt
from singa import device
from singa import tensor
from singa import sonnx
from singa import layer
from singa import autograd
import onnx

from singa_auto.model import BaseModel, utils
from singa_auto.constants import ModelDependency
from singa_auto.model.dev import make_predictions_json, _check_model_class, _print_header, _check_dependencies, inform_user
from singa_auto.model.utils import load_model_class
from singa_auto.advisor.constants import Proposal, ParamsType

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


class MyModel(sonnx.SONNXModel):

    def __init__(self, onnx_model):
        super(MyModel, self).__init__(onnx_model)

    def forward(self, *x):
        y = super(MyModel, self).forward(*x)
        return y[0]

    def train_one_batch(self, x, y):
        pass


class OnnxTinyYoloV2(BaseModel):

    def __init__(self, device_id=0, **knobs):
        super().__init__(**knobs)
        # onnx model url
        self.model_url = 'https://onnxzoo.blob.core.windows.net/models/opset_8/tiny_yolov2/tiny_yolov2.tar.gz'

        # model path in the downloaded tar file
        self.model_path = 'tiny_yolov2/Model.onnx'
        self.dev = device.create_cuda_gpu_on(device_id)

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

    def predict(self, queries: List[PIL.Image.Image]):
        print("Get queries")

        res = []
        queries = [self._preprocess(ele) for ele in queries]

        for img in queries:
            x = tensor.Tensor(device=self.dev, data=img)
            y = self._model.forward(x)
            result = self._postprocess(tensor.to_numpy(y)[0])
            res.append(result)
        return res

    def load_parameters(self, params):
        self._model = self._build_model()
        x = tensor.PlaceHolder((1, 3, 416, 416), device=self.dev)
        self._model.compile([x], is_train=False,
                            use_graph=True, sequential=True)

    def _build_model(self):
        # read and make onnx model
        download_model(self.model_url)
        onnx_model = onnx.load(os.path.join(
            '/tmp', self.model_path))
        model = MyModel(onnx_model)
        return model

    @staticmethod
    def _preprocess(query: PIL.Image.Image):
        img = query.resize((416, 416))
        img = np.array(img).astype(np.float32)
        img = np.rollaxis(img, 2, 0)
        img = np.expand_dims(img, axis=0)
        return img

    @staticmethod
    def _postprocess(out: List):
        numClasses = 20
        anchors = [1.08, 1.19, 3.42, 4.41, 6.63,
                   11.38, 9.42, 5.11, 16.62, 10.52]

        def sigmoid(x, derivative=False):
            return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))

        def softmax(x):
            scoreMatExp = np.exp(np.asarray(x))
            return scoreMatExp / scoreMatExp.sum(0)

        clut = [(0, 0, 0), (255, 0, 0), (255, 0, 255), (0, 0, 255), (0, 255, 0),
                (0, 255, 128), (128, 255, 0), (128, 128, 0), (0, 128, 255),
                (128, 0, 128), (255, 0, 128), (128, 0, 255), (255, 128, 128),
                (128, 255, 128), (255, 255, 0), (255, 128, 128), (128, 128, 255),
                (255, 128, 128), (128, 255, 128)]
        label = [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
            "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]

        result = dict()
        result['status'] = "ok"
        result['predictions'] = []

        for cy in range(13):
            for cx in range(13):
                for b in range(5):
                    channel = b * (numClasses + 5)
                    tx = out[channel][cy][cx]
                    ty = out[channel + 1][cy][cx]
                    tw = out[channel + 2][cy][cx]
                    th = out[channel + 3][cy][cx]
                    tc = out[channel + 4][cy][cx]
                    x = (float(cx) + sigmoid(tx)) * 32
                    y = (float(cy) + sigmoid(ty)) * 32

                    w = np.exp(tw) * 32 * anchors[2 * b]
                    h = np.exp(th) * 32 * anchors[2 * b + 1]
                    width, height = 416, 416
                    confidence = sigmoid(tc)

                    classes = np.zeros(numClasses)
                    for c in range(0, numClasses):
                        classes[c] = out[channel + 5 + c][cy][cx]

                    classes = softmax(classes)
                    detectedClass = classes.argmax()
                    if 0.5 < classes[detectedClass] * confidence:
                        left = x - w / 2
                        upper = y - h / 2
                        right = x + w / 2
                        down = y + h / 2
                        jbox = dict()
                        jbox['label_id'] = str(detectedClass)
                        jbox['label'] = str(label[detectedClass])
                        jbox['probability'] = classes[detectedClass] * confidence

                        jbox['detection_box'] = [max(0, upper / height), max(0, left / width),
                                                 min(1, down / height), min(1, right / width)]

                        result['predictions'].append(jbox)

        return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--query_path',
        type=str,
        default='examples/data/object_detection/test_person.jpg',
        help='Path(s) to query image(s), delimited by commas')
    (args, _) = parser.parse_known_args()

    queries = utils.dataset.load_images(args.query_path.split(','))
    
    model_file_path = __file__
    model_class = 'OnnxTinyYoloV2'
    dependencies = {ModelDependency.SINGA: '3.0.0',
                    ModelDependency.ONNX: '1.15.0'}
    task = 'IMAGE_OBJECT_DETECTION'

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

    (predictions, model_inst) = make_predictions_json(queries, task,
                                                 py_model_class,
                                                 proposal,
                                                 fine_tune_dataset_path=None,
                                                 params={})

    py_model_class.teardown()

    inform_user('No errors encountered while testing model!')
