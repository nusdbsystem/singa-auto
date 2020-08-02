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
import copy
import io
import json
import random
import PIL
import cv2
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
import base64
import tempfile
import numpy as np
from io import BytesIO
import torch
import torchvision
import logging
from typing import List
from singa_auto.datasets.coco_eval import CocoEvaluator
from singa_auto.datasets.coco_utils import get_coco_api_from_dataset
from singa_auto.datasets.torch_utils import get_transform
from singa_auto.model import ObjtDetModel, utils


logger = logging.getLogger(__name__)


class SaMaskRcnn(ObjtDetModel):
    '''
    Implements a maskrcnn
    '''

    @staticmethod
    def get_knob_config():
        return {}

    def __collate_fn(self, batch):
            return tuple(zip(*batch))

    def __init__(self):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("self.device", self.device)
        logger.info(self.device)

        self.model = None

        self.dataset_name = None

        # default is person , only one class
        self.filter_classes = ["person"]

    def train(self, dataset_path, **kwargs):

        logger.info("Training params: {}".format(json.dumps(kwargs)))

        num_classes = kwargs["num_classes"] if "num_classes" in kwargs else 2
        num_epoch = kwargs["num_epoch"] if "num_epoch" in kwargs else 10
        self.dataset_name = kwargs["dataset_name"] if "dataset_name" in kwargs else "pennfudan"
        batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 2

        if "filter_classes" in kwargs:
            self.filter_classes = kwargs["filter_classes"]

        print(self.filter_classes)
        annotation_dataset_path = kwargs['annotation_dataset_path']

        dataset_train = utils.dataset.load_img_detection_datasets(
            dataset_path=dataset_path,
            dataset_name=self.dataset_name,
            filter_classes=self.filter_classes,
            annotation_dataset_path=annotation_dataset_path,
            is_train=True)
        print("Total training images : ", len(dataset_train))
        # only extract train zip file once, so copy train instance, and share the same variables
        dataset_val = copy.copy(dataset_train)
        dataset_val.transforms = get_transform(is_train=False)

        logger.info("Training the model MaskRcnn using {}".format(self.device))
        print("Training the model MaskRcnn using {}".format(self.device))

        # split the dataset in train and val set
        indices = torch.randperm(len(dataset_train)).tolist()
        dataset_train = torch.utils.data.Subset(dataset_train, indices[:-4])
        dataset_val = torch.utils.data.Subset(dataset_val, indices[-4:])

        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True, collate_fn=self.__collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
                    dataset_val, batch_size=1, shuffle=False, collate_fn=self.__collate_fn)

        # get the model using our helper function
        self.model = self._build_model(num_classes)

        # move model to the right device
        self.model.to(self.device)

        # construct an optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params,
                                    lr=0.005,
                                    momentum=0.9,
                                    weight_decay=0.0005)
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

        for epoch in range(num_epoch):
            # train for one epoch, printing every 10 iterations
            loss_value = self._train_one_epoch(self.model, optimizer, data_loader, self.device, epoch)

            logger.info("loss is {}".format(loss_value))
            print("loss is {}".format(loss_value))

            if loss_value is None:
                break
            # update the learning rate
            lr_scheduler.step()

            logger.info("begin to evalute after epoch: {}".format(epoch))
            _, evaluate_res_str = self._evaluate(data_loader_test)

            logger.info("evalute after epoch {}, result is:".format(epoch))
            print(evaluate_res_str)
            logger.info(evaluate_res_str)

    def _train_one_epoch(self, model, optimizer, data_loader, device, epoch):
        model.train()

        lr_scheduler = None
        if epoch == 0:
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(data_loader) - 1)

            lr_scheduler = self.__warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

        logger.info("On Epoch {}, begin to train".format(epoch))
        loss_value = 0

        batch_num = 0
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            loss_value = losses.item()

            logger.info("Epoch: {}, Batch: {}, Loss: {}".format(epoch, batch_num, loss_value))
            print("Epoch: {}, Batch: {}, Loss: {}".format(epoch, batch_num, loss_value))

            if not np.math.isfinite(loss_value):
                logger.info("Loss is {}, stopping training".format(loss_value))
                return None

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()
            batch_num += 1

        return loss_value

    def dump_parameters(self):
        params = {}
        with tempfile.NamedTemporaryFile() as tmp:
            # Save whole model to temp h5 file
            torch.save(self.model, tmp.name)
            # Read from temp h5 file & encode it to base64 string
            with open(tmp.name, 'rb') as f:
                weight_base64 = f.read()
        params['weight_base64'] = base64.b64encode(weight_base64).decode('utf-8')
        params["filter_classes"] = json.dumps(self.filter_classes)
        return params

    def load_parameters(self, params):

        self.filter_classes = json.loads(params["filter_classes"])
        weight_base64 = params['weight_base64']

        weight_base64_bytes = base64.b64decode(weight_base64.encode('utf-8'))

        self.model = torch.load(BytesIO(weight_base64_bytes), map_location=self.device)

    def evaluate(self, dataset_path, **kwargs):
        print(kwargs)
        print(self.filter_classes)
        annotation_dataset_path = kwargs['annotation_dataset_path']

        dataset_test = utils.dataset.load_img_detection_datasets(
            dataset_path=dataset_path,
            dataset_name=self.dataset_name,
            filter_classes=self.filter_classes,
            annotation_dataset_path=annotation_dataset_path,
            is_train=False)

        data_loader_test = torch.utils.data.DataLoader(
                    dataset_test, batch_size=1, shuffle=False, collate_fn=self.__collate_fn)

        score, evaluate_res_str = self._evaluate(data_loader_test)
        logger.info("Evaluate id done, reuslt is : , score is {}".format(score))
        logger.info(evaluate_res_str)
        return score

    @torch.no_grad()
    def _evaluate(self, data_loader):
        cpu_device = torch.device("cpu")
        self.model.eval()

        coco = get_coco_api_from_dataset(data_loader.dataset)
        iou_types = self.__get_iou_types(self.model)
        coco_evaluator = CocoEvaluator(coco, iou_types)

        for images, targets in data_loader:
            images = list(img.to(self.device) for img in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            outputs = self.model(images)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            coco_evaluator.update(res)

        coco_evaluator.synchronize_between_processes()

        coco_evaluator.accumulate()
        evaluate_res_dict, evaluate_res_str = coco_evaluator.summarize()

        score = 0.5 * (
                evaluate_res_dict["bbox"]["Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]"]
                + evaluate_res_dict["segm"]["Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]"])

        return score, evaluate_res_str

    def predict(self, queries: List[PIL.Image.Image]) -> List[dict]:

        result = list()

        for img in queries:
            img_res = dict()
            img = np.asarray(img).astype(np.uint8)
            res = self.__get_prediction(img, threshold=0.8)
            if res is None:
                img_with_box = img_with_segmentation = img
            else:
                masks, boxes, pred_cls = res
                img_with_box = self.__get_bounding_box(img, masks, boxes, pred_cls)
                img_with_segmentation = self.__get_segmentation(img, masks)

            # the response format is only used to show on origin web ui
            img_res['explanations'] = {}
            img_res['explanations']['lime_img'] = self.__convert_img_to_str(img_with_box)
            img_res['explanations']['gradcam_img'] = self.__convert_img_to_str(img_with_segmentation)
            img_res['mc_dropout'] = []
            result.append(img_res)
        return result

    def _build_model(self, num_classes):
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
        utils.logger.log("Begin to load the maskrcnn model")
        # load an instance segmentation model pre-trained pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           num_classes)

        return model

    def __warmup_lr_scheduler(self, optimizer, warmup_iters, warmup_factor):

        def f(x):
            if x >= warmup_iters:
                return 1
            alpha = float(x) / warmup_iters
            return warmup_factor * (1 - alpha) + alpha

        return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

    def __get_prediction(self, img, threshold):
        img = F.to_tensor(img)
        img = img.to(self.device)
        pred = self.model([img])
        pred = [{ele: value.cpu() for ele, value in pred[0].items()}]
        pred_score = list(pred[0]['scores'].detach().cpu().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold]
        if not pred_t:
            return None
        pred_t = pred_t[-1]
        masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
        logger.info(list(pred[0]['labels'].numpy()))
        pred_class = [self.filter_classes[i-1] for i in list(pred[0]['labels'].numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
        masks = masks[:pred_t + 1]
        pred_boxes = pred_boxes[:pred_t + 1]
        pred_class = pred_class[:pred_t + 1]
        return masks, pred_boxes, pred_class

    def __get_bounding_box(self, img, masks, boxes, pred_cls,  rect_th=3, text_size=1, text_th=3):
        """
        draw the bounding box on img
        """
        img = copy.deepcopy(img)
        for i in range(len(masks)):
            cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)
            cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0),
                        thickness=text_th)

        return img

    def __get_segmentation(self, img, masks):
        """
        draw the segmentation box on img
        """
        def random_colour_masks(image):
            """
            for display the prediction image
            """
            colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255],
                       [80, 70, 180], [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
            r = np.zeros_like(image).astype(np.uint8)
            g = np.zeros_like(image).astype(np.uint8)
            b = np.zeros_like(image).astype(np.uint8)
            r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0, 10)]
            coloured_mask = np.stack([r, g, b], axis=2)
            return coloured_mask

        img = copy.deepcopy(img)
        for i in range(len(masks)):
            rgb_mask = random_colour_masks(masks[i])
            img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        return img

    def __convert_img_to_str(self, arr):
        im = Image.fromarray(arr.astype("uint8"))
        rawBytes = io.BytesIO()
        encoding = 'utf-8'
        im.save(rawBytes, "PNG")
        rawBytes.seek(0)
        return base64.b64encode(rawBytes.read()).decode(encoding)

    def __get_iou_types(self, model):
        model_without_ddp = model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_without_ddp = model.module
        iou_types = ["bbox"]
        if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
            iou_types.append("segm")
        if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
            iou_types.append("keypoints")
        return iou_types


if __name__ == "__main__":
    import argparse
    from singa_auto.model.dev import test_model_class
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path',
                        type=str,
                        default='/train2017.zip',
                        help='Path to train dataset')
    parser.add_argument('--val_path',
                        type=str,
                        default='/hdd1/val2017.zip',
                        help='Path to validation dataset')

    parser.add_argument('--annotation_dataset_path',
                        type=str,
                        default='/hdd1/annotations_trainval2017.zip',
                        help='Path to validation dataset')

    # parser.add_argument('--test_path',
    #                     type=str,
    #                     default='/hdd1/PennFudanPed.zip',
    #                     help='Path to test dataset')
    parser.add_argument(
        '--query_path',
        type=str,
        default='examples/data/object_detection/person_cat.jpeg',
        help='Path(s) to query image(s), delimited by commas')

    (args, _) = parser.parse_known_args()

    print(args.query_path.split(','))

    queries = utils.dataset.load_images(args.query_path.split(','))
    test_model_class(model_file_path=__file__,
                     model_class='SaMaskRcnn',
                     task='IMAGE_DETECTION',
                     dependencies={"torch": "1.4.0+cu100",
                                   "torchvision": "0.5.0+cu100",
                                   "opencv-python": "4.2.0.34",
                                   "pycocotools": "2.0.1"},
                     train_dataset_path=args.train_path,
                     val_dataset_path=args.val_path,
                     annotation_dataset_path=args.annotation_dataset_path,
                     test_dataset_path=None,
                     train_args={"num_classes": 3,
                                 "num_epoch": 1,
                                 "dataset_name": "coco2017",
                                 "filter_classes": ["car", 'cat'],
                                 # "dataset_name": "pennfudan",
                                 "batch_size": 2
                                 },
                     queries=queries)

    """
    Test the model out of singa-auto platform
    python -c "import torch;print(torch.cuda.is_available())"
    """

    # model = SaMaskRcnn()
    # model_file = "person_cat_weight.model"
    # with open(model_file, 'rb') as f:
    #     content = f.read()
    #
    # weight_base64 = base64.b64encode(content).decode('utf-8')
    # params = {}
    # params['weight_base64'] = weight_base64
    #
    # from singa_auto.param_store import FileParamStore
    # params = FileParamStore("/").load(model_file)
    # model.load_parameters(params)
    #
    # with open("examples/data/object_detection/person_cat.jpeg", 'rb') as f:
    #     img_bytes = [f.read()]
    # queries = utils.dataset.load_images(img_bytes)
    # model.predict(queries)

