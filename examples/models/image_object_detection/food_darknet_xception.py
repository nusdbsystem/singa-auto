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


import base64
import io
import os
import tempfile
import zipfile
from typing import List

import numpy as np
from PIL import Image

from keras.models import Model
from keras.layers import Dense
from keras.preprocessing.image import img_to_array
from keras.applications.xception import Xception
from rafiki.model import BaseModel


class FoodDetection(BaseModel):
    '''
    Implements Xception on Keras for IMAGE_CLASSIFICATION
    '''

    def __init__(self, **knobs):
        super().__init__(**knobs)
        self.xception_model = None
        self.det_net = None
        self.det_meta = None
        self.classes = 231
        self.image_size = 299
        self.class_dict = {}

    def train(self, dataset_path, **kwargs):
        pass

    def get_knob_config(self):
        pass

    def evaluate(self, dataset_path):
        pass

    def destroy(self):
        pass

    def dump_parameters(self):
        pass

    def predict(self, queries):
        res = []
        queries = [self.image_to_byte_array(ele) for ele in queries]

        for img_bytes in queries:
            with tempfile.NamedTemporaryFile() as tmp:
                with open(tmp.name, 'wb') as f:
                    f.write(img_bytes)
                img_path = tmp.name
                img = Image.open(img_path)
                width, height = img.size[0], img.size[1]
                predications = self._recognition(img_path)

                result = dict()
                result['status'] = "ok"
                result['predictions'] = []

                for index, box in enumerate(predications):
                    prob = box[1]
                    x, y, w, h = box[2][0], box[2][1], box[2][2], box[2][3]
                    left = x - w / 2
                    upper = y - h / 2
                    right = x + w / 2
                    down = y + h / 2
                    cropped = img.crop((x - w / 2, y - h / 2, x + w / 2, y + h / 2))  # (left, upper, right, lower)
                    y = self._classify(cropped)

                    class_id = np.argsort(y[0])[::-1][0]
                    str_class = self.class_dict[class_id]
                    jbox = dict()
                    jbox['label_id'] = str(class_id)
                    jbox['label'] = str(str_class)
                    jbox['probability'] = prob
                    # y_min,x_min,y_max,x_max

                    jbox['detection_box'] = [max(0, upper / height), max(0, left / width),
                                             min(1, down / height), min(1, right / width)]

                    result['predictions'].append(jbox)

                res.append(result)
        return res

    def load_parameters(self, params):

        self.class_dict = {v: k for k, v in np.load("rafiki/custom_models_base/darknet/cfg/food/food231.npy")[()].items()}

        h5_models_base64 = params['h5_model_base64']

        self.xception_model = self._build_model(classes=self.classes, image_size=self.image_size)

        with tempfile.NamedTemporaryFile() as tmp:
            # Convert back to bytes & write to temp file
            h5_models_bytes = base64.b64decode(h5_models_base64.encode('utf-8'))
            with open(tmp.name, 'wb') as f:
                f.write(h5_models_bytes)
            with tempfile.TemporaryDirectory() as d:
                dataset_zipfile = zipfile.ZipFile(tmp.name, 'r')
                dataset_zipfile.extractall(path=d)
                dataset_zipfile.close()

                for file_name in os.listdir(d):
                    if "yolo" in file_name:
                        self.det_net = darknet.load_net(b".yolov3-food.cfg",
                                                        os.path.join(d, file_name).encode(), 0)

                        self.det_meta = darknet.load_meta(b".food.data")

                    if "xception" in file_name:
                        self.xception_model.load_weights(os.path.join(d, file_name))

    def _build_model(self, classes, image_size):
        base_model = Xception(include_top=True, input_shape=(image_size, image_size, 3))
        base_model.layers.pop()
        predictions = Dense(classes, activation='softmax')(base_model.layers[-1].output)
        clf_model = Model(input=base_model.input, output=[predictions])
        return clf_model

    def _recognition(self, img_path):
        res = darknet.detect(self.det_net, self.det_meta, str.encode(img_path))
        return res

    def _classify(self, img):
        width_height_tuple = (self.image_size, self.image_size)
        if (img.size != width_height_tuple):
            img = img.resize(width_height_tuple, Image.NEAREST)
        x = img_to_array(img)
        x /= 255 * 1.
        x = x.reshape((1,) + x.shape)
        y = self.xception_model.predict(x)
        return y

    @staticmethod
    def image_to_byte_array(query: List[str]):
        query = np.asarray(query).astype(np.uint8)
        image = Image.fromarray(query)
        imgByteArr = io.BytesIO()
        image.save(imgByteArr, format='JPEG')
        imgByteArr = imgByteArr.getvalue()
        return imgByteArr
