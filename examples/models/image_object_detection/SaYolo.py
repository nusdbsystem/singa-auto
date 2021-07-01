import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import sys
sys.path.append(os.getcwd())

import base64
import copy
import cv2
import io
import json
import logging
import numpy as np
import random
import tempfile
import torch
import torchvision
import zipfile

import PIL

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from terminaltables import AsciiTable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from typing import List

# Singa-auto Dependency
from singa_auto.darknet.model import DarkNet
from singa_auto.darknet.utils import ap_per_class
from singa_auto.darknet.utils import get_batch_statistics
from singa_auto.darknet.utils import non_max_suppression
from singa_auto.darknet.utils import pad_to_square
from singa_auto.darknet.utils import rescale_boxes
from singa_auto.darknet.utils import resize
from singa_auto.darknet.utils import weights_init_normal
from singa_auto.darknet.utils import xywh2xyxy
from singa_auto.datasets.image_detection_dataset import YoloDataset
from singa_auto.datasets.image_detection_dataset import fetch_from_train_set
from singa_auto.datasets.image_detection_dataset import split_dataset
from singa_auto.model.dev import test_model_class
from singa_auto.model.knob import FixedKnob
# from singa_auto.model.model import BaseModel
from singa_auto.model.object_detection import ObjtDetModel
from singa_auto.model.utils import utils


logger = logging.getLogger(__name__)


class SaYolo(ObjtDetModel):
    """
    implements a yolo
    """
    def __init__(self, **knobs):
        super().__init__(**knobs)
        self._knobs = knobs

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("using device", self.device)
        logger.info("using device", self.device)

        self.model = None
        self.dataset_name = None
        self.gradient_accumulations = 2

        # default is cat , only one class
        self.filter_classes = ['cat']

        # # make sure results folder exists
        # if os.path.exists(r"./results/"):
        #     import shutil
        #     shutil.rmtree(r"./results/")
        # os.makedirs(r"./results/")
    
    @staticmethod
    def get_knob_config():
        return {
            "conf_thresh": FixedKnob(0.1),
            "lr": FixedKnob(0.01),
            "model_def": FixedKnob("./singa_auto/darknet/yolov3-tiny.cfg"),
            "nms_thresh": FixedKnob(0.2),
            "pretrained_weights": FixedKnob("./singa_auto/darknet/darknet53.conv.74"),
        }

    def is_predict_valid(self, box_info, class_info, image_size):
        """
        make sure predicted result is valid, ie coordinates and labels are correct
        """
        if box_info[6] - 1 in range(len(class_info)) and min(box_info[0:4]) >= 0 and max(box_info[0:4]) < image_size:
            return True
        else:
            return False

    def __collate_fn(self, batch):
        return tuple(zip(*batch))

    def train(self, dataset_path, **kwargs):
        logger.info("Training params: {}".format(json.dumps(kwargs)))

        # num_classes = len(self._knobs.get("filter_classes"))
        num_epoch = kwargs["num_epoch"] if "num_epoch" in kwargs else 2
        batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 8

        if "filter_classes" in kwargs:
            self.filter_classes = kwargs["filter_classes"]

        print("{} in train.".format(self.filter_classes))
        logger.info("{} in train.".format(self.filter_classes))

        # root_path = r"/home/taomingyang/dataset/coco_mini_cat/"

        # load 
        dataset_zipfile = zipfile.ZipFile(dataset_path, 'r')
        train_folder = tempfile.TemporaryDirectory()
        dataset_zipfile.extractall(path=train_folder.name)
        root_path = train_folder.name
        print("root_path: {}".format(root_path))
        logger.info("root_path: {}".format(root_path))

        print("prepare dataset")
        logger.info("prepare dataset")
        if os.path.isdir(os.path.join(root_path, "image")):
            print("split train/val subsets...")
            logger.info("split train/val subsets...")
            split_dataset(root_path)          
        elif os.path.isdir(os.path.join(root_path, "train")):
            if not os.path.exists(os.path.join(root_path, "val")):
                logger.info("fetch val from train")
                fetch_from_train_set(root_path)
        else:
            print("unsupported dataset format!")
            logger.info("unsupported dataset format!")
            return None

        image_train = os.path.join(root_path, "train", "image")
        image_val = os.path.join(root_path, "val", "image")
        annotation_train = os.path.join(root_path, "train", "annotation")
        annotation_val = os.path.join(root_path, "val", "annotation")

        # Get dataloader
        dataset_train = YoloDataset(
            image_train,
            annotation_train,
            is_single_json_file=False,
            filter_classes=self.filter_classes,
            is_train=True,
            augment=True,
            multiscale=True
        )
        # Get dataloader
        dataset_test = YoloDataset(
            image_val,
            annotation_val,
            is_single_json_file=False,
            filter_classes=self.filter_classes,
            is_train=False,
            augment=False,
            multiscale=False
        )

        print("Training the model YOLO using {}".format(self.device))
        logger.info("Training the model YOLO using {}".format(self.device))

        # define training and validation data loaders
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True, collate_fn=dataset_train.collate_fn
        )

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=batch_size, shuffle=False, collate_fn=dataset_test.collate_fn
        )

        # get the model using our helper function
        self.model = DarkNet(config_path=self._knobs.get("model_def")).to(self.device)
        self.model.apply(weights_init_normal)

        # pretrained weights
        if self._knobs.get("pretrained_weights"):
            pretrained_weights_path = self._knobs.get("pretrained_weights")
            if pretrained_weights_path.endswith(".pth"):
                if os.path.exists(pretrained_weights_path):
                    self.model.load_state_dict(torch.load(pretrained_weights_path, map_location="cpu"))
                    logger.info("using pretrained_weights {}".format(pretrained_weights_path))
                else:
                    logger.warning("pretrained_weights {} not exists.".format(pretrained_weights_path))
            else:
                if not os.path.exists(pretrained_weights_path):
                    import wget
                    os.makedirs(os.path.dirname(pretrained_weights_path), exist_ok=True)
                    pretrained_weights_path = wget.download(r"https://pjreddie.com/media/files/darknet53.conv.74", out=os.path.dirname(pretrained_weights_path))
                self.model.load_darknet_weights(pretrained_weights_path)
                logger.info("using pretrained_weights {}".format(pretrained_weights_path))

        # # move model to the right device
        # self.model.to(self.device)

        # construct an optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self._knobs.get("lr"),
        )
        
        torch.manual_seed(1)

        for epoch in range(num_epoch):
            # train for one epoch, printing every 10 iterations
            loss_value = self._train_one_epoch(self.model, optimizer, data_loader_train, epoch)

            print("loss is {}".format(loss_value))
            logger.info("loss is {}".format(loss_value))

            if loss_value is None:
                break

            # update the learning rate
            # lr_scheduler.step()

            print("begin to evalute after epoch: {}".format(epoch))
            logger.info("begin to evalute after epoch: {}".format(epoch))
            precision, recall, AP, f1, ap_class = self._evaluate(data_loader_test)
            print("Average Precisions:")
            logger.info("Average Precisions:")
            for i, c in enumerate(ap_class):
                info_str = "\t+ Class \"{}\" ({}) - AP: {:.5f}".format(c, dataset_test.coco.cats[dataset_test.label_to_cat[c]]['name'], AP[i])
                print(info_str)
                logger.info(info_str)
            print("mAP: {:.9f}".format(AP.mean()))
            logger.info("mAP: {:.9f}".format(AP.mean()))

    def _train_one_epoch(self, model, optimizer, data_loader, epoch):
        model.train()

        # lr_scheduler = None
        # if epoch == 0:
        #     warmup_factor = 1. / 1000
        #     warmup_iters = min(1000, len(data_loader) - 1)
        # 
        #     lr_scdheduler = self.__warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

        logger.info("On Epoch {}, begin to train".format(epoch))
        # loss_value = 0

        metrics = [
            "grid_size",
            "loss",
            "x",
            "y",
            "w",
            "h",
            "conf",
            "cls",
            "cls_acc",
            "recall50",
            "recall75",
            "precision",
            "conf_obj",
            "conf_noobj",
        ]

        for batch_i, (_, images, targets) in enumerate(data_loader):
            logger.info("\t batch {}/{}".format(batch_i, len(data_loader)))
            batches_done = len(data_loader) * epoch + batch_i
            
            images = images.to(self.device)
            targets = targets.to(self.device)

            loss, outputs = model(images, targets)

            if not np.math.isfinite(loss):
                logger.info("Loss is {}, stopping training".format(loss))
                return None

            loss.backward()

            if batches_done % self.gradient_accumulations:
                optimizer.step()
                optimizer.zero_grad()

            # log_str = "\n---- [Epoch %d, Batch %d/%d] ----\n" % (epoch, batch_i, len(data_loader))
            # 
            # metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(self.model.yolo_layers))]]]
            # 
            # # Log metrics at each YOLO layer
            # for i, metric in enumerate(metrics):
            #     formats = {m: "%.6f" for m in metrics}
            #     formats["grid_size"] = "%2d"
            #     formats["cls_acc"] = "%.2f%%"
            #     row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in self.model.yolo_layers]
            #     metric_table += [[metric, *row_metrics]]
            # 
            #     # # Tensorboard logging
            #     # tensorboard_log = []
            #     # for j, yolo in enumerate(model.yolo_layers):
            #     #     for name, metric in yolo.metrics.items():
            #     #         if name != "grid_size":
            #     #             tensorboard_log += [(f"{name}_{j+1}", metric)]
            #     # tensorboard_log += [("loss", loss.item())]
            #     # summary_logger.list_of_scalars_summary(tensorboard_log, batches_done)
            # 
            # log_str += AsciiTable(metric_table).table
            # log_str += f"\nTotal loss {loss.item()}"
            # print(log_str)
            # logger.info(log_str)
            # 
            # if lr_scheduler is not None:
            #     lr_scheduler.step()

            model.seen += images.size(0)

        return loss.item()

    def dump_parameters(self):
        """
        dump parameters to local file
        """
        params = {}
        with tempfile.NamedTemporaryFile() as tmp:
            # Save whole model to temp h5 file
            torch.save(self.model.state_dict(), tmp.name)
            # Read from temp h5 file & encode it to base64 string
            with open(tmp.name, 'rb') as f:
                weight_base64 = f.read()
        params['weight_base64'] = base64.b64encode(weight_base64).decode('utf-8')
        params["module_cfg"] = json.dumps(self.model.model_cfg)
        return params

    def load_parameters(self, params):
        """
        load parameters from local file
        """

        logger.info("load parameters")
        weight_base64 = params['weight_base64']
        self.module_cfg = json.loads(params["module_cfg"])

        weight_base64_bytes = base64.b64decode(weight_base64.encode('utf-8'))

        state_dict = torch.load(io.BytesIO(weight_base64_bytes), map_location=self.device)
        self.model = DarkNet(model_cfg=self.module_cfg).to(self.device)
        self.model.load_state_dict(state_dict)
        # self.model.cuda()

    def evaluate(self, dataset_path, **kwargs):
        print(kwargs)

        # root_path = r"/home/taomingyang/dataset/coco_mini_cat/"

        # load 
        dataset_zipfile = zipfile.ZipFile(dataset_path, 'r')
        evaluate_folder = tempfile.TemporaryDirectory()
        dataset_zipfile.extractall(path=evaluate_folder.name)
        root_path = evaluate_folder.name
        print(root_path)
        logger.info("root_path: {}".format(root_path))

        print("prepare dataset")
        if os.path.isdir(os.path.join(root_path, "image")):
            print("split train/val subsets...")
            logger.info("split train/val subsets...")
            split_dataset(root_path)
        elif os.path.isdir(os.path.join(root_path, "train")):
            if not os.path.exists(os.path.join(root_path, "val")):
                fetch_from_train_set(root_path)
                logger.info("fetch val from train")
        else:
            print("unsupported dataset format!")
            logger.info("unsupported dataset format!")
            return None

        image_val = os.path.join(root_path, "val", "image")
        annotation_val = os.path.join(root_path, "val", "annotation")

        dataset_valid = YoloDataset(
            image_val,
            annotation_val,
            is_single_json_file=False,
            filter_classes=self.filter_classes,
            is_train=False,
        )
        data_loader_valid = torch.utils.data.DataLoader(
            dataset_valid,
            batch_size=1,
            shuffle=False,
            collate_fn=dataset_valid.collate_fn
        )

        logger.info("dataset prepared")

        # perform an evaluate
        precision, recall, AP, f1, ap_class = self._evaluate(data_loader_valid)
        
        return np.mean(precision)

    @torch.no_grad()
    def _evaluate(self, data_loader):
        self.model.eval()

        labels = []
        sample_metrics = []  # List of tuples (TP, confs, pred)

        for batch_i, (names, images, targets) in enumerate(data_loader):
            # Extract labels
            labels += targets[:, 1].tolist()
            # Rescale target
            targets[:, 2:] = xywh2xyxy(targets[:, 2:])
            targets[:, 2:] *= 416

            images = images.to(self.device)

            with torch.no_grad():
                outputs = self.model(images)
                outputs = non_max_suppression(outputs, conf_thresh=self._knobs.get("conf_thresh"), nms_thresh=self._knobs.get("nms_thresh"))
            
            for name, image, output in zip(names, images, outputs):
                tmp = image.cpu().detach().permute((1, 2, 0)).mul(255).clamp(0, 255).numpy()
                tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR)
                if output is not None:
                    for rect_info in output:
                        coord = rect_info.cpu().numpy()
                        if self.is_predict_valid(coord, self.filter_classes, image.size(-1)):
                            cv2.rectangle(tmp, (coord[0], coord[1]), (coord[2], coord[3]), (0, 255, 0), 3)
            
                cv2.imwrite('./results/{}'.format(os.path.basename(name)), tmp)

            sample_metrics += get_batch_statistics(outputs, targets, iou_thresh=0.5)

        # return score, evaluate_res_str
        if 0 == len(sample_metrics):
            ap_class = np.array(list(set(labels)), dtype=np.int32)
            precision = recall = AP = f1 = np.array([0 for x in ap_class], dtype=np.float64)
        else:
            # Concatenate sample statistics
            true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
            precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

        return precision, recall, AP, f1, ap_class

    def predict(self, queries: List[PIL.Image.Image]) -> List[dict]:
        """
        predict with trained model
        """

        os.makedirs("./results/", exist_ok=True)

        result = list()

        for img in queries:
            img_res = dict()

            if isinstance(img, List):
                print(len(img))
                img = np.array(img[0])
                img_data = Image.fromarray(np.uint8(img))
            elif isinstance(img, np.ndarray):
                img_data = Image.fromarray(img)
            else:
                img_data = img

            # get prediction
            res = self.__get_prediction(img_data)
            if res is None:
                img_with_box = img_with_segmentation = img_data
                boxes, pred_cls = None, None
            else:
                boxes, pred_cls = res
                img_data = np.asarray(img_data).astype(np.uint8)
                img_with_box = self.__get_bounding_box(img_data, boxes, pred_cls)

            # the response format is only used to show on origin web ui
            img_res['explanations'] = {}
            # img_res['explanations']['lime_img'] = self.__convert_img_to_str(img_with_box)
            # img_res['explanations']['box_info'] = boxes
            # img_res['explanations']['classes'] = pred_cls
            
            img_res['explanations']['box_info'] = []
            
            if boxes is not None and pred_cls is not None and len(boxes) == len(pred_cls):
                for box_coord, class_name in zip(boxes, pred_cls):
                    img_res['explanations']['box_info'].append({
                        "coord": box_coord,
                        "class_name": class_name,
                    })
            img_res['mc_dropout'] = []
            
            result.append(img_res)

        return result

    def __warmup_lr_scheduler(self, optimizer, warmup_iters, warmup_factor):
        def f(x):
            if x >= warmup_iters:
                return 1
            alpha = float(x) / warmup_iters
            return warmup_factor * (1 - alpha) + alpha

        return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

    def __get_prediction(self, img):
        self.model.eval()
        
        img = torchvision.transforms.ToTensor()(img)
        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1], img.shape[2]))
        elif len(img.shape) == 3 and img.shape[0] == 1:
            img = img.expand((3, img.shape[1], img.shape[2]))

        ori_size = img.shape[-2:]

        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        img = torch.unsqueeze(resize(img, 416), 0)
        img = img.to(self.device)
        pred = self.model(img)
        pred = non_max_suppression(pred, conf_thresh=self._knobs.get("conf_thresh"), nms_thresh=self._knobs.get("nms_thresh"))
        pred_class = []
        pred_boxes = []
        if pred[0] is None:
            return None
        
        box_info = rescale_boxes(pred[0], 416, ori_size)
        num_box = box_info.size()[0]

        # get predicted info
        for rect_info in box_info:
            coord = rect_info.cpu().numpy()
            if self.is_predict_valid(coord, self.filter_classes, img.size(-1)):
                pred_class.append(self.filter_classes[np.int(coord[6])-1])
                pred_boxes.append((np.int(coord[0]), np.int(coord[1]), np.int(coord[2]), np.int(coord[3])))
        
        if len(pred_boxes) == 0:
            return None
        else:
            return pred_boxes, pred_class

    def __get_bounding_box(self, img, boxes, pred_cls,  rect_th=3, text_size=1, text_th=3):
        """
        draw the bounding box on img
        """
        
        img = copy.deepcopy(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for i in range(len(boxes)):
            cv2.rectangle(img, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (0, 255, 0), rect_th)
            cv2.putText(img, pred_cls[i], (boxes[i][0], boxes[i][1]), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0),
                        thickness=text_th)

        cv2.imwrite("./results/{:04d}.png".format(random.randint(0, 9999)), img)
        return img

    def __get_segmentation(self, img, masks):
        """
        draw the segmentation box on img
        """
        def random_colour_masks(image):
            """
            for display the prediction image
            """
            colours = [
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [0, 255, 255],
                [255, 255, 0],
                [255, 0, 255],
                [80, 70, 180],
                [250, 80, 190],
                [245, 145, 50],
                [70, 150, 250],
                [50, 190, 190]
            ]
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
    parser.add_argument(
        '--train_path',
        type=str,
        default='/home/taomingyang/dataset/package/coco_mini.zip',
        help='Path to train dataset'
    )
    parser.add_argument(
        '--val_path',
        type=str,
        default='/home/taomingyang/dataset/package/coco_mini.zip',
        help='Path to validation dataset'
    )
    parser.add_argument(
        '--query_path',
        type=str,
        default='./examples/data/object_detection/cat.jpg',
        help='Path(s) to query image(s), delimited by commas'
    )

    (args, _) = parser.parse_known_args()

    queries = utils.dataset.load_images(args.query_path.split(','))
    test_model_class(
        model_file_path=__file__,
        model_class='SaYolo',
        task='OBJECT_DETECTION',
        dependencies={
            "opencv-python": "4.4.0.46",
            "terminaltables": "3.1.0",
            "torch": "1.4.0",
            "torchvision": "0.5.0",
        },
        train_dataset_path=args.train_path,
        val_dataset_path=args.val_path,
        train_args={
            "batch_size": 8,
            "model_def": "./singa_auto/darknet/yolov3-tiny.cfg",
            "filter_classes": ['cat'],
            "num_epoch": 1,
            "pretrained_weights": "./singa_auto/darknet/darknet53.conv.74",
        },
        queries=queries
    )

    """
    Test the model out of singa-auto platform
    python -c "import torch;print(torch.cuda.is_available())"
    """

