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
import tempfile
from rafiki.model import BaseModel, utils
from PIL import Image
from keras.models import Model
from keras.layers import Dense, K
from keras.preprocessing.image import img_to_array
from keras.applications.xception import Xception
import numpy as np
from typing import List, Any


class AutoXception(BaseModel):
    '''
    Implements Xception on Keras for IMAGE_CLASSIFICATION
    '''

    def __init__(self, **knobs):
        super().__init__(**knobs)
        self._model = None
        self.classes = 231
        self.image_size = 299

    def get_knob_config(self):
        pass

    def train(self, dataset_path, **kwargs):
        pass

    def evaluate(self, dataset_path):
        pass

    def destroy(self):
        pass

    def dump_parameters(self):
        pass

    def predict(self, queries: List[Any]):
        print('start prediction in predict method')
        res = list()
        width_height_tuple = (self.image_size, self.image_size)
        try:
            for img_list in queries:
                img = self._to_pil_img(img_list)
                if img.size != width_height_tuple:
                    img = img.resize(width_height_tuple, Image.NEAREST)
                x = img_to_array(img)
                x /= 255 * 1.
                x = x.reshape((1,) + x.shape)
                y = self._model.predict(x)
                res.append(y.squeeze().tolist())
        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()
        print(res)
        return res

    def load_parameters(self, params):

        h5_model_base64 = params['h5_model_base64']

        with tempfile.NamedTemporaryFile() as tmp:
            # Convert back to bytes & write to temp file
            h5_model_bytes = base64.b64decode(h5_model_base64.encode('utf-8'))
            with open(tmp.name, 'wb') as f:
                f.write(h5_model_bytes)
                self._model = self._build_model(classes=self.classes, image_size=self.image_size)
                self._model.load_weights(tmp.name)

    def _build_model(self, classes, image_size):
        base_model = Xception(include_top=True, input_shape=(image_size, image_size, 3))
        base_model.layers.pop()
        predictions = Dense(classes, activation='softmax')(base_model.layers[-1].output)
        clf_model = Model(input=base_model.input, output=[predictions])
        return clf_model

    @staticmethod
    def _to_pil_img(query: List[str]) -> Image:
        query = np.asarray(query).astype(np.uint8)
        image = Image.fromarray(query)
        return image
