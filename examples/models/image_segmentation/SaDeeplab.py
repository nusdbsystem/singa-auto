import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2, 3"

import sys
sys.path.append(os.getcwd())

import base64
import json
import logging
import tempfile
import zipfile
from collections.abc import Sequence
from copy import deepcopy
from io import BytesIO
from typing import List
from glob import glob
from time import time
import requests
from singa_auto.model import BaseModel, CategoricalKnob, FixedKnob, utils
from singa_auto.model.knob import BaseKnob


import tensorflow as tf

tf.random.set_seed(100)

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import random
import cv2
from PIL import Image
import h5py
from tensorflow.python.keras.saving import hdf5_format

class OxfordPets(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + (self.img_size, self.img_size) + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=(self.img_size, self.img_size))
            x[j] = img
        # y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        # for j, path in enumerate(batch_target_img_paths):
        #     img = load_img(path, target_size=self.img_size, color_mode="grayscale")
        #     y[j] = np.expand_dims(img, 2)
        #     # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
        #     y[j] -= 1
        
        y = np.zeros((self.batch_size,) + (self.img_size * self.img_size,) + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=(self.img_size, self.img_size), color_mode="grayscale")
            img = np.array(img).flatten()
            y[j] = np.expand_dims(img, 1)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2: (already update dataset, this operation has expired)
            # y[j] -= 1
        
        sample_weight = np.zeros((self.batch_size,) + (self.img_size * self.img_size,), dtype="float32")
        for k in range(self.batch_size):
            unique_class = np.unique(y)
            if len(unique_class):
                class_weights = {class_id: 1.0 for class_id in unique_class}
                class_weights[unique_class[-1]] = 0.0
            for yy in unique_class:
                np.putmask(sample_weight[k], y[k]==yy, class_weights[yy])
            np.putmask(sample_weight[k], y[k]==unique_class[-1], class_weights[unique_class[-1]])

        return x, y, sample_weight


def Bottleneck(input_shape, output_channels, stride=1, dilation=1):
    '''
    a classic residual convolution module
    '''
    inputs = tf.keras.Input(input_shape)
    residual  = inputs

    # residual conv branch
    x = layers.Conv2D(output_channels, (1, 1), padding='same', use_bias=False)(residual)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(output_channels, (3, 3), strides=(stride, stride), padding='same', 
                        dilation_rate=(dilation, dilation), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(output_channels * 4, (1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # original branch
    if stride != 1 or inputs.shape[-1] != x.shape[-1]:
        residual = layers.Conv2D(output_channels * 4, (1, 1), padding='same', 
                                strides=(stride, stride), use_bias=False)(residual)
        residual = layers.BatchNormalization()(residual)

    # merge two branches
    x = layers.Add()([x, residual])
    x = layers.ReLU()(x)

    # export model
    return keras.Model(inputs=inputs, outputs=x)


def ResNetAtrous(layer_num=[3, 4, 6, 3], dilations=[1, 2, 1]):
    '''
    an atrous conv version resnet50 model
    '''
    inputs = keras.Input((None, None, 3))
    strides = [2, 1, 1]

    # conv
    x = layers.Conv2D(64, (7, 7), (2, 2), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x, training=False)
    x = layers.ReLU()(x)

    # down-sampling
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # resblock 1
    for i in range(layer_num[0]):
        x = Bottleneck(x.shape[1:], 64, stride=1, dilation=1)(x)
    low = x # save low level features

    # resblock 2
    for i in range(layer_num[1]):
        x = Bottleneck(x.shape[1:], 128, stride=strides[0] if i == 0 else 1, dilation=1)(x) 

    # resblock 3
    for i in range(layer_num[2]):
        x = Bottleneck(x.shape[1:], 256, stride=strides[1] if i == 0 else 1, dilation=1)(x)

    # resblock 4
    for i in range(layer_num[3]):
        x = Bottleneck(x.shape[1:], 512, stride=strides[2] if i == 0 else 1, dilation=dilations[i])(x)
    high = x

    return keras.Model(inputs=inputs, outputs=(low, high))


def ASPP(input_channels):
    inputs = layers.Input((None, None, input_channels))

    # global pooling
    global_mean = layers.Lambda(lambda x: tf.math.reduce_mean(x, [1, 2], keepdims=True))(inputs) # size (b, 1, 1, c)
    
    global_mean = layers.Conv2D(256, (1, 1), padding='same', 
                                kernel_initializer=tf.keras.initializers.he_normal(), use_bias=False)(global_mean)
    global_mean = layers.BatchNormalization()(global_mean)
    global_mean = layers.ReLU()(global_mean) # size (b, 1, 1, 256)

    global_mean = layers.Lambda(lambda x: tf.image.resize(x[0], (tf.shape(x[1])[1], tf.shape(x[1])[2]), 
                            method=tf.image.ResizeMethod.BILINEAR))([global_mean, inputs]) # size (b, h, w, 256)
    
    # dilation with rate 1
    dilated_1 = layers.Conv2D(256, (1, 1), dilation_rate=1, padding='same', 
                                kernel_initializer=tf.keras.initializers.he_normal(), use_bias=False)(inputs)
    dilated_1 = layers.BatchNormalization()(dilated_1)
    dilated_1 = layers.ReLU()(dilated_1)

    # dilation with rate 6
    dilated_6 = layers.Conv2D(256, (3, 3), dilation_rate=6, padding='same', 
                                kernel_initializer=tf.keras.initializers.he_normal(), use_bias=False)(inputs)
    dilated_6 = layers.BatchNormalization()(dilated_6)
    dilated_6 = layers.ReLU()(dilated_6)

    # dilation with rate 12
    dilated_12 = layers.Conv2D(256, (3, 3), dilation_rate=12, padding='same', 
                                kernel_initializer=tf.keras.initializers.he_normal(), use_bias=False)(inputs)
    dilated_12 = layers.BatchNormalization()(dilated_12)
    dilated_12 = layers.ReLU()(dilated_12)

    # dilation with rate 18
    dilated_18 = layers.Conv2D(256, (3, 3), dilation_rate=18, padding='same', 
                                kernel_initializer=tf.keras.initializers.he_normal(), use_bias=False)(inputs)
    dilated_18 = layers.BatchNormalization()(dilated_18)
    dilated_18 = layers.ReLU()(dilated_18)

    # concate pyramid
    x = layers.Concatenate(axis=-1)([global_mean, dilated_1, dilated_6, dilated_12, dilated_18])
    x = layers.Conv2D(256, (1, 1), padding='same', 
                        kernel_initializer=tf.keras.initializers.he_normal(), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return keras.Model(inputs=inputs, outputs=x)


def DeepLabV3Plus(img_size, n_classes):
    inputs = keras.Input(shape=img_size + (3,))
    # inputs = keras.Input((None, None, 3))

    low, high = ResNetAtrous(layer_num=[3, 4, 6, 3], dilations=[1, 2, 1])(inputs)

    # modify low level feature channel number
    low = layers.Conv2D(48, (1, 1), padding='same', 
                        kernel_initializer=keras.initializers.he_normal(), use_bias=False)(low)
    low = layers.BatchNormalization()(low)
    low = layers.ReLU()(low) # size (b, h/4, w/4, 48)

    # pass high level feature into ASPP module
    high = ASPP(high.shape[-1])(high) # size (b, h/8, w/8, 256)
    high = layers.Lambda(lambda x: tf.image.resize(x[0], (tf.shape(x[1])[1], tf.shape(x[1])[2]), 
                        method = tf.image.ResizeMethod.BILINEAR))([high, low]);
    # concate and modify channel
    x = layers.Concatenate(axis=-1)([high, low]) # size (b, h/4, w/4, 304)

    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', 
                        kernel_initializer=keras.initializers.he_normal(), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', 
                        kernel_initializer=keras.initializers.he_normal(), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # upsampling
    x = layers.Lambda(lambda x: tf.image.resize(x[0], (tf.shape(x[1])[1], tf.shape(x[1])[2]), 
                        method = tf.image.ResizeMethod.BILINEAR))([x, inputs])

    
    # full conv
    x = layers.Conv2D(n_classes, (1,1), padding='same', activation=keras.activations.softmax, 
                        name = 'full_conv')(x)

    # flatten
    x = layers.Reshape((img_size[0] * img_size[1], n_classes))(x)


    return keras.Model(inputs=inputs, outputs=x)


class CustomModel(keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compiled_loss(
                y,
                y_pred,
                sample_weight=sample_weight,
                regularization_losses=self.losses,
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


logger = logging.getLogger(__name__)


class SaDeeplab(BaseModel):
    '''
    train deeplab
    '''
    @staticmethod
    def get_knob_config():
        return {
            # hyper parameters
            "lr": FixedKnob(1e-4),
            "batch_size": FixedKnob(2),
            "epoch": FixedKnob(1),

            # application parameters
            # "num_classes": FixedKnob(1),
            "fine_size": FixedKnob(160),
            "train_val_split_rate": FixedKnob(0.9),

        }


    def __init__(self, **knobs):
        super().__init__(**knobs)
        self._knobs = knobs

        self.model = None

        self.fine_size = self._knobs.get("fine_size")
        self.split_rate = self._knobs.get("train_val_split_rate")


    def train(self, dataset_path, **kwargs):
        # hyper parameters 
        self.batch_size = self._knobs.get("batch_size")
        self.epoch = self._knobs.get("epoch")

        self.lr = self._knobs.get("lr")
 

        logger.info("Training params: {}".format(json.dumps(kwargs)))


        # extract uploaded zipfile
        dataset_zipfile = zipfile.ZipFile(dataset_path, 'r')

        train_folder = tempfile.TemporaryDirectory()
        folder_name = train_folder.name
        dataset_zipfile.extractall(path=folder_name)

        # load train params from zipfile
        with open(os.path.join(folder_name, 'param.json'),'r') as load_f:
            load_dict = json.load(load_f)
            self.num_classes = load_dict["num_classes"] if "num_classes" in list(load_dict.keys()) else 21 # default class number(21) is the same as voc2012
            print(f"total number of classes: {self.num_classes}")
            logger.info(f"total number of classes: {self.num_classes}")

        # load images from zipfile
        if os.path.isdir(os.path.join(folder_name, "image")):
            print("split train/val subsets...")
            logger.info("split train/val subsets...")

            # load image and mask seperately
            input_img_paths = sorted(
                [
                    os.path.join(folder_name, "image", fname)
                    for fname in os.listdir(os.path.join(folder_name, "image"))
                ]
            )
            target_img_paths = sorted(
                [
                    os.path.join(folder_name, "mask", fname)
                    for fname in os.listdir(os.path.join(folder_name, "mask"))
                ]
            )
            self.num_image = len(input_img_paths)
            print("Total image number: ", self.num_image) 
            logger.info(f"Total image number : {self.num_image}")

            # split train/val
            val_samples = int((1 - self.split_rate) * self.num_image)
            # random.Random(1337).shuffle(input_img_paths)
            # random.Random(1337).shuffle(target_img_paths)
            train_input_img_paths = input_img_paths[:-val_samples]
            train_target_img_paths = target_img_paths[:-val_samples]
            val_input_img_paths = input_img_paths[-val_samples:]
            val_target_img_paths = target_img_paths[-val_samples:]
            
            print(f"train images: {len(train_input_img_paths)}, val images: {len(val_input_img_paths)}") 
            logger.info(f"train images: {len(train_input_img_paths)}, val images: {len(val_input_img_paths)}")  

        elif os.path.isdir(os.path.join(folder_name, "train")):
            print("directly load train/val datasets...")
            logger.info("directly load train/val datasets...")

            # load image and mask seperately
            train_input_img_paths = sorted(
                [
                    os.path.join(folder_name, "train", "image", fname)
                    for fname in os.listdir(os.path.join(folder_name, "train", "image"))
                ]
            )
            train_target_img_paths = sorted(
                [
                    os.path.join(folder_name, "train", "mask", fname)
                    for fname in os.listdir(os.path.join(folder_name, "train", "mask"))
                ]
            )

            val_input_img_paths = sorted(
                [
                    os.path.join(folder_name, "val", "image", fname)
                    for fname in os.listdir(os.path.join(folder_name, "val", "image"))
                ]
            )
            val_target_img_paths = sorted(
                [
                    os.path.join(folder_name, "val", "mask", fname)
                    for fname in os.listdir(os.path.join(folder_name, "val", "mask"))
                ]
            )
            self.num_image = len(train_input_img_paths) + len(val_input_img_paths)
            print("Total image number: ", self.num_image) 
            logger.info(f"Total image number : {self.num_image}")

            print(f"train images: {len(train_input_img_paths)}, val images: {len(val_input_img_paths)}") 
            logger.info(f"train images: {len(train_input_img_paths)}, val images: {len(val_input_img_paths)}")  
        else:
            print("unsupported dataset format!")
            logger.info("unsupported dataset format!")

        # load dataset
        train_gen = OxfordPets(
                self.batch_size, self.fine_size, train_input_img_paths, train_target_img_paths
        )
        val_gen = OxfordPets(self.batch_size, self.fine_size, val_input_img_paths, val_target_img_paths)

        logger.info("Training the model DeeplabV3+ using {}".format("cuda" if tf.test.is_gpu_available()==True else "cpu"))
        print("Training the model DeeplabV3+ using {}".format("cuda" if tf.test.is_gpu_available()==True else "cpu"))

        # clear session buffer 
        keras.backend.clear_session()

        # get the model using our helper function
        inputs = keras.Input(shape=(self.fine_size, self.fine_size) + (3,))
        outputs = DeepLabV3Plus((self.fine_size, self.fine_size), self.num_classes)(inputs)
        self.model = CustomModel(inputs, outputs)
        self.model.summary()

        # coimpile model with optimizers...
        self.model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
        callbacks = [
            # keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # start training
        self.model.fit(train_gen, epochs=self.epoch, validation_data=val_gen, callbacks=callbacks)
    

    def evaluate(self, val_dataset_path, **kwargs):
        # extract validation datasets
        dataset_zipfile = zipfile.ZipFile(val_dataset_path, 'r')
        val_folder = tempfile.TemporaryDirectory()
        dataset_zipfile.extractall(path=val_folder.name)
        folder_name = val_folder.name

        if os.path.isdir(os.path.join(folder_name, "image")):
            print("split train/val subsets...")
            logger.info("split train/val subsets...")

            # load image and mask seperately
            input_img_paths = sorted(
                [
                    os.path.join(folder_name, "image", fname)
                    for fname in os.listdir(os.path.join(folder_name, "image"))
                ]
            )
            target_img_paths = sorted(
                [
                    os.path.join(folder_name, "mask", fname)
                    for fname in os.listdir(os.path.join(folder_name, "mask"))
                ]
            )

            # split train/val
            val_samples = int((1 - self.split_rate) * self.num_image)
            # random.Random(1337).shuffle(input_img_paths)
            # random.Random(1337).shuffle(target_img_paths)
            train_input_img_paths = input_img_paths[:-val_samples]
            train_target_img_paths = target_img_paths[:-val_samples]
            val_input_img_paths = input_img_paths[-val_samples:]
            val_target_img_paths = target_img_paths[-val_samples:]

        elif os.path.isdir(os.path.join(folder_name, "train")):
            print("directly load train/val datasets...")
            logger.info("directly load train/val datasets...")

            # load image and mask seperately
            train_input_img_paths = sorted(
                [
                    os.path.join(folder_name, "train", "image", fname)
                    for fname in os.listdir(os.path.join(folder_name, "train", "image"))
                ]
            )
            train_target_img_paths = sorted(
                [
                    os.path.join(folder_name, "train", "mask", fname)
                    for fname in os.listdir(os.path.join(folder_name, "train", "mask"))
                ]
            )

            val_input_img_paths = sorted(
                [
                    os.path.join(folder_name, "val", "image", fname)
                    for fname in os.listdir(os.path.join(folder_name, "val", "image"))
                ]
            )
            val_target_img_paths = sorted(
                [
                    os.path.join(folder_name, "val", "mask", fname)
                    for fname in os.listdir(os.path.join(folder_name, "val", "mask"))
                ]
            )

        else:
            print("unsupported dataset format!")
            logger.info("unsupported dataset format!")

        val_gen = OxfordPets(self.batch_size, self.fine_size, val_input_img_paths, val_target_img_paths)

        loss, accuracy = self.model.evaluate(val_gen)

        
        return accuracy


    def dump_parameters(self):
        params = {}
        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            
            # Save whole model to a tempfile
            self.model.save_weights(tmp.name)
            # Read from tempfile & encode it to base64 string
            # with h5py.File(tmp.name, 'r') as f:
            #     if 'layer_names' not in f.attrs and 'model_weights' in f:
            #         weights_h5 = f['model_weights']
            with open(tmp.name, 'rb') as f:
                weight_base64 = f.read()
        params['weight_base64'] = base64.b64encode(weight_base64).decode('utf-8')
        params['num_classes'] = self.num_classes
        return params


    def load_parameters(self, params):
        weight_base64 = params['weight_base64']
        self.num_classes = params['num_classes']

        weight_base64_bytes = base64.b64decode(weight_base64.encode('utf-8'))

        # state_dict = torch.load(BytesIO(weight_base64_bytes), map_location=self.device)
        
        inputs = keras.Input(shape=(self.fine_size, self.fine_size) + (3,))
        outputs = DeepLabV3Plus((self.fine_size, self.fine_size), self.num_classes)(inputs)
        self.model = CustomModel(inputs, outputs)

        # weight_h5 = h5py.File(BytesIO(weight_base64_bytes))
        with h5py.File(BytesIO(weight_base64_bytes), 'r') as f:
            hdf5_format.load_weights_from_hdf5_group(f, self.model.layers)

        # self.model.load_weights(weight_h5)


    def _get_prediction(self, img):

        image = cv2.resize(img.astype('float32'), (self.fine_size, self.fine_size))
        print("+"*30)
        print(image.shape)
        image = np.expand_dims(image, axis=0)
        predict = self.model.predict(image)

        mask = np.argmax(predict, axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        mask = np.reshape(mask, (160, 160))

        h, w = image.shape[:2]

        # transform result image into original size
        mask_out = cv2.resize(mask.astype(np.uint8), (w, h), cv2.INTER_NEAREST)


        return mask_out


    def predict(self, queries: List[List]) -> List[dict]:
        # print(len(queries))
        result = list()

        # depending on different input types, need different conditions
        for idx, img in enumerate(queries):
            logger.info(type(img))
            if isinstance(img, List):
                print(len(img))
                img = np.array(img[0])
                print(img.shape)
                img_file = img
                # print(type(img_file))
            elif isinstance(img, Image.Image):
                img_file = np.array(img)
            else:
                img_data = img

            # get prediction
            res_raw = self._get_prediction(img_file)

            # add color palette (we follow the VOC2012 color map and the max num_class is 21)
            res_raw = res_raw.astype(np.uint8)
            res = Image.fromarray(res_raw)
            palette = []
            for i in range(256):
                palette.extend((i, i, i))
            palette[:3*21] = np.array([[0, 0, 0],
                                [128, 0, 0],
                                [0, 128, 0],
                                [128, 128, 0],
                                [0, 0, 128],
                                [128, 0, 128],
                                [0, 128, 128],
                                [128, 128, 128],
                                [64, 0, 0],
                                [192, 0, 0],
                                [64, 128, 0],
                                [192, 128, 0],
                                [64, 0, 128],
                                [192, 0, 128],
                                [64, 128, 128],
                                [192, 128, 128],
                                [0, 64, 0],
                                [128, 64, 0],
                                [0, 192, 0],
                                [128, 192, 0],
                                [0, 64, 128]
                             ], dtype='uint8').flatten()
            res.putpalette(palette)

            name = f"./query_{idx}.png"
            res.save(name)
            full_name = os.path.abspath(name)

            buffered = BytesIO()
            res.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue())


            result.append(img_str.decode('utf-8'))
            
            # result.append(requests.get('http://192.168.100.203:36667/fetch').text)
                
        return result

if __name__ == "__main__":
    import argparse

    from singa_auto.model.dev import test_model_class
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path',
                        type=str,
                        default='./dataset/oxford_pets/datasets.zip',
                        help='Path to train dataset')
    parser.add_argument('--val_path',
                        type=str,
                        default='./dataset/oxford_pets/datasets.zip',
                        help='Path to validation dataset')

    # parser.add_argument('--annotation_dataset_path',
    #                     type=str,
    #                     default='./dataset/voc2012/val2014.zip',
    #                     help='Path to validation dataset')

    # parser.add_argument('--test_path',
    #                     type=str,
    #                     default='/hdd1/PennFudanPed.zip',
    #                     help='Path to test dataset')
    parser.add_argument(
        '--query_path',
        type=str,
        default='/home/zhaozixiao/projects/singa_local/singa-auto/dataset/oxford_pets/Persian_120.jpg,/home/zhaozixiao/projects/singa_local/singa-auto/dataset/oxford_pets/pomeranian_159.jpg',
        help='Path(s) to query image(s), delimited by commas'
    )

    (args, _) = parser.parse_known_args()

    # print(args.query_path.split(','))

    imgs = utils.dataset.load_images(args.query_path.split(','))
    img_nps = []
    for i in imgs:
        img = np.array(i)
        img_nps.append(img)
    
    queries = img_nps
    test_model_class(model_file_path=__file__,
                     model_class='SaDeeplab',
                     task='IMAGE_SEGMENTATION',
                     dependencies={
                        "tensorflow": "2.3.0",
                        "opencv": "3.4.2.16",
                    },
                     train_dataset_path=args.train_path,
                     val_dataset_path=args.val_path,
                     test_dataset_path=None,
                     train_args={"num_classes": 3},
                     queries=img_nps)



    











