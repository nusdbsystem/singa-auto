import abc
import base64
import io
import os
import tempfile
import zipfile
from typing import List

import PIL
import numpy as np
from PIL import Image

from keras.models import Model
from keras.layers import Dense
from keras.preprocessing.image import img_to_array

from singa_auto.darknet import darknet
from singa_auto.model import BaseModel


class FoodDetectionBase(BaseModel):

    def __init__(self, clf_model_class_name, **knobs):
        super().__init__(**knobs)
        # model
        # this is the model class we use
        self.clf_model_class_name = clf_model_class_name

        # this is the model after we build
        self.clf_model = None

        # this is the darknet model
        self.det_net = None
        self.det_meta = None

        # labels
        self.class_dict = {}

        # pre config
        self.classes = None
        self.image_size = None

        # preload files
        self.yolo_cfg_name = None
        self.yolo_weight_name = None
        self.food_name = None

        # this is the model file downloaded from internet,
        # can choose download locally and upload , or download from server
        # if download at server side, leave it to none
        self.preload_clf_model_weights_name = None

        # this is the trained model
        self.trained_clf_model_weights_name = None

        self._npy_index_name = None

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

    def predict(self, queries: List[PIL.Image.Image]) -> List[dict]:
        print("Get queries")

        res = []
        queries = [self.image_to_byte_array(ele) for ele in queries]

        for img_bytes in queries:
            with tempfile.NamedTemporaryFile() as tmp:
                with open(tmp.name, 'wb') as f:
                    f.write(img_bytes)
                img_path = tmp.name
                img = Image.open(img_path)
                width, height = img.size[0], img.size[1]
                predications = self._detection(img_path)

                result = dict()
                result['status'] = "ok"
                result['predictions'] = []
                print("Detection is done, begin to do the classification")
                for index, box in enumerate(predications):
                    prob = box[1]
                    x, y, w, h = box[2][0], box[2][1], box[2][2], box[2][3]
                    left = x - w / 2
                    upper = y - h / 2
                    right = x + w / 2
                    down = y + h / 2
                    # (left, upper, right, lower)
                    cropped = img.crop((x - w / 2, y - h / 2, x + w / 2, y + h / 2))
                    y = self._classification(cropped)

                    class_id = np.argsort(y[0])[::-1][0]
                    str_class = self.class_dict[class_id]
                    jbox = dict()
                    jbox['label_id'] = str(class_id)
                    jbox['label'] = str(str_class)
                    jbox['probability'] = prob

                    jbox['detection_box'] = [max(0, upper / height), max(0, left / width),
                                             min(1, down / height), min(1, right / width)]

                    result['predictions'].append(jbox)

                res.append(result)
        return res

    def load_parameters(self, params):

        # get the zip file bytes
        zip_file_base64 = params['zip_file_base64']

        with tempfile.NamedTemporaryFile() as tmp:

            # Convert back to bytes & write to temp file
            zip_file_base64_bytes = base64.b64decode(zip_file_base64.encode('utf-8'))

            # write the bytes to local file
            with open(tmp.name, 'wb') as f:
                f.write(zip_file_base64_bytes)

            # extract the zip file
            with tempfile.TemporaryDirectory() as root_path:
                dataset_zipfile = zipfile.ZipFile(tmp.name, 'r')
                dataset_zipfile.extractall(path=root_path)
                dataset_zipfile.close()

                print("Begin to load model dependences")

                # generate yolo dependence pathes
                yolo_cfg_path = os.path.join(root_path, self.yolo_cfg_name)
                yolo_weight_path = os.path.join(root_path, self.yolo_weight_name)
                food_name = os.path.join(root_path, self.food_name)

                # generate a food.data file, which is used by darknet
                food_data_path = os.path.join(root_path, "food.data")

                with open(food_data_path, 'wb') as f:
                    f.write(
                        "classes= 1\ntrain  = '""'\nvalid  = '""'\nnames = {}"
                            .format(food_name)
                            .encode()
                    )

                if self.preload_clf_model_weights_name:
                    preload_clf_model_weight_path = os.path.join(root_path, self.preload_clf_model_weights_name)
                else:
                    preload_clf_model_weight_path = None

                trained_clf_model_weights_path = os.path.join(root_path, self.trained_clf_model_weights_name)

                npy_index_path = os.path.join(root_path, self._npy_index_name)

                # load model files for darknet
                self.det_net = darknet.load_net(yolo_cfg_path.encode(),
                                                yolo_weight_path.encode(),
                                                0)

                self.det_meta = darknet.load_meta(food_data_path.encode())

                print("Begin to build models")

                # load pre-trained model for classification model
                self.clf_model = self._build_model(
                    weight_path=preload_clf_model_weight_path,
                    classes=self.classes,
                    image_size=self.image_size)

                # load custom trained model for classification model
                self.clf_model.load_weights(trained_clf_model_weights_path)

                self.class_dict = {v: k for k, v in np.load(npy_index_path)[()].items()}

        print("Loading params...Done!")

    def _build_model(self, weight_path, classes, image_size):
        if weight_path:
            base_model = self.clf_model_class_name(
               weights=weight_path,
               include_top=True,
               input_shape=(image_size, image_size, 3)
               )
        else:
            base_model = self.clf_model_class_name(
               include_top=True,
               input_shape=(image_size, image_size, 3)
               )
        base_model.layers.pop()
        predictions = Dense(classes, activation='softmax')(base_model.layers[-1].output)
        clf_model = Model(input=base_model.input, output=[predictions])
        return clf_model

    def _detection(self, img_path):
        res = darknet.detect(self.det_net, self.det_meta, str.encode(img_path))
        return res

    def _classification(self, img):
        width_height_tuple = (self.image_size, self.image_size)
        if (img.size != width_height_tuple):
            img = img.resize(width_height_tuple, Image.NEAREST)
        x = img_to_array(img)
        x /= 255 * 1.
        x = x.reshape((1,) + x.shape)
        y = self.clf_model.predict(x)
        return y

    @staticmethod
    def image_to_byte_array(query: PIL.Image.Image):
        query = np.asarray(query).astype(np.uint8)
        image = Image.fromarray(query)
        imgByteArr = io.BytesIO()
        image.save(imgByteArr, format='JPEG')
        imgByteArr = imgByteArr.getvalue()
        return imgByteArr

