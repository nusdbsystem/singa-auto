import os
from os.path import join
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import sys
sys.path.append(os.getcwd())

import time
import json
import base64
import random
import logging
import zipfile
import tempfile
import datetime
import requests
from PIL import Image
from io import BytesIO
from typing import List
from collections import OrderedDict

from singa_auto.model import ObjtDetModel, FixedKnob
from singa_auto.model.dev import test_model_class
from singa_auto.model.utils import dataset

import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class FoodlgNet(ObjtDetModel):

    @staticmethod
    def get_knob_config():
        return {
            'learning_rate': FixedKnob(1e-10),
            'momentum': FixedKnob(0.7),
            'epoch': FixedKnob(0),
            'batch_size': FixedKnob(4)
        }

    def __init__(self, **knobs):
        # load model parameters and configurations
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # image preprocessing function
        self.cls_transform = transforms.Compose([
            transforms.Resize((299,299), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # initialize other parameters
        self.lr = knobs.get('learning_rate')
        self.momentum = knobs.get('momentum')
        self.epoch = knobs.get('epoch')
        self.batch_size = knobs.get('batch_size')

    def _initialize_model(self, paths=None):
        # two networks, yolo detection network (to train) and resnext classification network (fixed)

        if paths is None:
            # here you need to set an extra file server to provide config/pretrained weights files
            # the codes below will download files from file server.
            # During prediction, this process is not required any more.
            dst_folder = tempfile.TemporaryDirectory()
            dst_folder_path = dst_folder.name
            # object_names_path = load_url(save_path=dst_folder_path, url='http://192.168.100.203:8000/FoodlgNet/classes.names')
            model_config_path = load_url(save_path=dst_folder_path, url='http://192.168.100.203:8000/FoodlgNet/foodlg_yolo.cfg')
            pretrain_model_path = load_url(save_path=dst_folder_path, url='http://192.168.100.203:8000/FoodlgNet/yolov3_ckpt_SGD_94.pth')

            classify_names_path = load_url(save_path=dst_folder_path, url='http://192.168.100.203:8000/FoodlgNet/food783.name')
            classify_model_path = load_url(save_path=dst_folder_path, url='http://192.168.100.203:8000/FoodlgNet/resnext101_ckpt.pth')

        else:
            model_config_path = paths['model_config_path']
            pretrain_model_path = paths['pretrain_model_path']
            classify_model_path = paths['classify_model_path']
            classify_names_path = paths['classify_names_path']


        # initiate detection model
        self.conf_thres = 0.5
        self.nms_thres = 0.4
        self.img_size = 416
        with open(model_config_path, encoding = 'utf-8') as f:
            self.model_config_path = f.readlines()

        det_model = Darknet(model_config_path, img_size=self.img_size)
        det_model.load_state_dict(torch.load(pretrain_model_path, map_location='cpu'))
        det_model = det_model.to(self.device)
        det_model.eval()
        self.det_model = det_model
        self.det_classes = ['food']

        # initiate classification model
        self.num_classes = 783
        self.clf_classes = load_classes(classify_names_path)

        from torchvision.models.resnet import Bottleneck, ResNet
        clf_model = ResNet(block=Bottleneck, layers=[3, 4, 23, 3], groups=32, width_per_group=16)
        clf_model.fc = nn.Linear(2048, self.num_classes)

        ckpt = torch.load(classify_model_path, map_location='cpu')
        ckpt = OrderedDict({k.replace('module.', ''): v for k, v in ckpt.items()})
        clf_model.load_state_dict(ckpt)
        clf_model = clf_model.to(self.device)
        clf_model.eval()
        self.clf_model = clf_model


    def train(self, dataset_path, shared_params=None, **train_args):
        # fine-tune yolov3 model for detectection part
        self._initialize_model()

        # load and process data
        dataset_folder = load_zip(dataset_path)
        # split dataset and then save to txt files
        train_path, valid_path = split_dataset_save(dataset_folder, ratio=0.9)
        # load data to torch dataloader
        dataset = ListDataset(train_path, augment=True, multiscale=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )
        # other settings for training
        optimizer = torch.optim.SGD(self.det_model.parameters(), lr=self.lr, momentum=self.momentum)

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

        # start training
        start_epoch = 0
        for epoch in range(start_epoch, self.epoch):
            self.det_model.train()
            start_time = time.time()
            for batch_i, (_, imgs, targets) in enumerate(dataloader):

                batches_done = len(dataloader) * epoch + batch_i

                imgs = imgs.to(self.device)
                targets = targets.to(self.device)

                loss, outputs = self.det_model(imgs, targets)
                loss.backward()

                if batches_done % 2:
                    # Accumulates gradient before each step
                    optimizer.step()
                    optimizer.zero_grad()

                # ----------------
                #   Log progress
                # ----------------

                log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, self.epoch, batch_i, len(dataloader))
                metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(self.det_model.yolo_layers))]]]

                # Log metrics at each YOLO layer
                for i, metric in enumerate(metrics):
                    formats = {m: "%.6f" for m in metrics}
                    formats["grid_size"] = "%2d"
                    formats["cls_acc"] = "%.2f%%"
                    row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in self.det_model.yolo_layers]
                    metric_table += [[metric, *row_metrics]]

                if batch_i%50==0:
                    log_str += toAscii(metric_table)
                    log_str += f"\nTotal loss {loss.item()}"

                    # Determine approximate time left for epoch
                    epoch_batches_left = len(dataloader) - (batch_i + 1)
                    time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
                    log_str += f"\n---- ETA {time_left}"

                    print(log_str)

                self.det_model.seen += imgs.size(0)

            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = _evaluate(
                self.det_model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=self.conf_thres,
                nms_thres=self.nms_thres,
                img_size=self.img_size,
                batch_size=self.batch_size,
                device=self.device
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, self.det_classes[c], "%.5f" % AP[i]]]
            # print(AsciiTable(ap_table).table)
            print(toAscii(ap_table))
            print(f"---- mAP {AP.mean()}")


    def dump_parameters(self):
        params = {}
        # get models weights
        det_model_weights = self.det_model.state_dict()
        clf_model_weights = self.clf_model.state_dict()

        # convert model weights and configs to json format
        params['det_model_weights'] = serialize_state_dict(det_model_weights)
        params['clf_model_weights'] = serialize_state_dict(clf_model_weights)
        params['det_model_cfg'] = json.dumps(self.model_config_path)
        params['clf_classes'] = json.dumps(self.clf_classes)

        return params

    def evaluate(self, dataset_path, **kargs):
        # load and process data
        dataset_folder = load_zip(dataset_path)
        # split dataset and then save to txt files
        train_path, valid_path = split_dataset_save(dataset_folder, ratio=0.9)
        # evaluate process
        precision, recall, AP, f1, ap_class = _evaluate(
            self.det_model,
            path=valid_path,
            iou_thres=0.5,
            conf_thres=self.conf_thres,
            nms_thres=self.nms_thres,
            img_size=self.img_size,
            batch_size=self.batch_size,
            device=self.device
        )

        return f1[0]

    def load_parameters(self, params):
        paths = {}
        # prepare tmp file paths
        dst_folder = tempfile.TemporaryDirectory().name
        os.mkdir(dst_folder)

        model_config_path = os.path.join(dst_folder,'det_config.pth')
        pretrain_model_path = os.path.join(dst_folder, 'det_model.pth')
        classify_model_path = os.path.join(dst_folder, 'clf_model.names')
        classify_names_path = os.path.join(dst_folder, 'clf_class.cfg')

        # convert params to python object and save to tmp paths
        det_model_weights = deserialize_state_dict(params['det_model_weights'])
        clf_model_weights = deserialize_state_dict(params['clf_model_weights'])

        torch.save(det_model_weights, pretrain_model_path)
        torch.save(clf_model_weights, classify_model_path)

        with open(model_config_path,'w', encoding = 'utf-8') as f:
            f.writelines(json.loads(params['det_model_cfg']))

        with open(classify_names_path,'w', encoding = 'utf-8') as f:
            for name in json.loads(params['clf_classes']):
                f.write(name + '\n')

        # record these paths
        paths['model_config_path'] = model_config_path
        paths['pretrain_model_path'] = pretrain_model_path
        paths['classify_model_path'] = classify_model_path
        paths['classify_names_path'] = classify_names_path

        # initiate model
        self._initialize_model(paths=paths)


    def predict(self, queries):
        logger.info(f'the length of queries is {len(queries)}')
        # queries is a list of PIL.Image object
        ##########
        # yolo part
        input_imgs = []
        widths = []
        heights = []

        queries = unifyImageType(queries)
        for img in queries:
            w, h = img.size
            widths.append(w)
            heights.append(h)

            img = transforms.ToTensor()(img) # Extract image as PyTorch tensor
            img, _ = pad_to_square(img, 0) # Pad to square resolution
            img = resize(img, self.img_size) # Resize
            input_imgs.append(img.unsqueeze(0))

        input_imgs = torch.cat(input_imgs)
        input_imgs = input_imgs.to(self.device)


        # Get detections
        with torch.no_grad():
            detections = self.det_model(input_imgs)
            detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)

        cls_results = []
        for det_res, width, height, img in zip(detections, widths, heights, queries):

            if det_res is None: # no food detected, skip irv2 part
                cls_results.append({
                    'explanations': {
                        'box_info': []
                    },
                    'raw_preds': [],
                    'mc_dropout': [],  # not used
                })
                continue

            ##########
            # irv2 part
            # pass each detection to classification model

            cropped_imgs = []
            bbox_values = []
            confs = []
            predictions = []

            det_res = rescale_boxes(det_res, self.img_size, (height, width))
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in det_res:

                x1, y1, x2, y2 = list(map(lambda x: x.tolist(), (x1, y1, x2, y2)))
                cropped = img.crop((x1, y1, x2, y2))  # (left, upper, right, lower)

                cropped_imgs.append(cropped)
                bbox_values.append([x1, y1, x2, y2])
                confs.append(conf.cpu().item())

            cropped_imgs = [self.cls_transform(img) for img in cropped_imgs]
            test_dataloader = DataLoader(cropped_imgs, batch_size=self.batch_size, shuffle=False)

            for batch in test_dataloader:

                batch = batch.to(self.device)
                # parallelly batch prediction
                with torch.no_grad():
                    prediction = self.clf_model(batch)

                predictions.extend(p.cpu().numpy() for p in prediction)

            # post processing to make result compatible with different front ends
            result = {
                'explanations':{
                    'box_info': []
                },
                'raw_preds': [],
                'mc_dropout': [], # not used
            }
            for idx in range(len(predictions)):
                class_id = np.argsort(predictions[idx])[::-1][:1]
                str_class = ' '.join(self.clf_classes[i] for i in class_id)

                jbox = {}
                jbox['label_id'] = str(class_id[0])
                jbox['label'] = str(str_class)
                jbox['probability'] = confs[idx]

                x1, y1, x2, y2 = bbox_values[idx]
                jbox['detection_box'] = [
                    max(0, y1 / height),
                    max(0, x1 / width),
                    min(1, y2 / height),
                    min(1, x2 / width)
                ]

                exp_box = {}
                exp_box['coord'] = [int(x1), int(y1), int(x2), int(y2)]
                exp_box['class_name'] = str(str_class)

                result['explanations']['box_info'].append(exp_box)
                result['raw_preds'].append(jbox)

            cls_results.append(result)

        logger.info(f'Predict result: {cls_results}')
        return cls_results

def unifyImageType(imgs):
    # to check if the image is PIL.Image or numpy.ndarray
    # and convert all to PIL.Image
    results = []
    for img in imgs:
        if isinstance(img, List):
            # used for accepting image from forkcloud frontend
            img = np.uint8(np.array(img[0])) 
            results.append(Image.fromarray(img))
        elif isinstance(img, np.ndarray):
            results.append(Image.fromarray(img))
        else:
            results.append(img)
    return results


def load_zip(zip_path):
    logger.info(zip_path)
    # extract uploaded zipfile
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f'zip file {zip_path} does not exist')

    dst_folder = tempfile.TemporaryDirectory().name
    zip_data = zipfile.ZipFile(zip_path, 'r')
    zip_data.extractall(path=dst_folder)
    return dst_folder

def load_url(save_path, url):
    # download file and save
    res = requests.get(url, timeout=300)
    filename = join(save_path, url.split('/')[-1])
    with open(filename, 'wb') as f:
        f.write(res.content)
    return filename

def serialize_state_dict(state_dict):
    with tempfile.NamedTemporaryFile() as tmp:
        torch.save(state_dict, tmp.name)
        with open(tmp.name, 'rb') as f:
            weights = f.read()

    return base64.b64encode(weights).decode('utf-8')

def deserialize_state_dict(b64bytes):
    b64bytes = base64.b64decode(b64bytes.encode('utf-8'))
    state_dict = torch.load(BytesIO(b64bytes), map_location='cpu')

    return state_dict

def split_dataset_save(dataset_folder, ratio = 0.9):
    image_paths = os.listdir(join(dataset_folder, 'images'))
    image_paths = [join(dataset_folder, 'images', path) for path in image_paths]

    # split dataset
    train_size = round(len(image_paths) * ratio)
    valid_size = len(image_paths) - train_size
    random.seed(42)
    random_idx = random.sample(range(len(image_paths)), k=train_size)
    random_idx = sorted(random_idx)

    train_set = []
    valid_set = []
    idx_pointer = 0
    for i, path in enumerate(image_paths):
        if i == random_idx[idx_pointer]:
            train_set.append(path)
            idx_pointer += 1
        else:
            valid_set.append(path)

    train_file_path = join(dataset_folder, 'train.txt')
    valid_file_path = join(dataset_folder, 'valid.txt')
    with open(train_file_path, 'w') as f:
        for line in train_set:
            f.write(line + '\n')

    with open(valid_file_path, 'w') as f:
        for line in valid_set:
            f.write(line + '\n')

    return train_file_path, valid_file_path


def toAscii(data_list):
    # convert evaluate result list to string
    res = ''
    for line in data_list:
        line = [str(l) for l in line]
        line = ','.join(line) + '\n'
        res += line
    return res

def _evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, device):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(dataloader):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size


        with torch.no_grad():
            imgs = imgs.to(device)
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class




#############################
# YOLOv3 part

def to_cpu(tensor):
    return tensor.detach().cpu()

def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r", encoding = 'utf-8')
    names = fp.read().split("\n")[:-1]
    return names

def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r', encoding = 'utf-8')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['): # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics

def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes

def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output

def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):

    # ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    ByteTensor = torch.cuda.BoolTensor if pred_boxes.is_cuda else torch.BoolTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)


    # # note: solver multi gpu problem
    # target  = target[target.sum(dim=1) != 0]

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()

    # prevent index out of boundary
    gi = gi.clamp(0, nG - 1)
    gj = gj.clamp(0, nG - 1)

    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())

        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_size = int(hyperparams["height"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                x, layer_loss = module[0](x, targets, img_dim)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            # path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            # just remove the postfix '.txt'
            path.replace("/images/", "/labels/").rstrip() + '.txt'
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)


if __name__ == "__main__":
    os.environ['WORKDIR_PATH'] = os.getcwd()
    os.environ['PARAMS_DIR_PATH'] = os.getcwd()
    
    test_img_paths = [
        '/home/jiahua/food_all/三明治/99_sanmingzhi.jpg',
        '/home/jiahua/food_all/三明治/0_sanmingzhi.jpg'
    ]
    imgs = dataset.load_images(test_img_paths)

    # # forkcloud format test
    # imgs = [[np.array(imgs[0]).tolist()]]

    test_model_class(model_file_path=__file__,
                     model_class='FoodlgNet',
                     task='IMAGE_DETECTION',
                     dependencies={},
                     train_dataset_path='/home/jiahua/singa-local/dataset.zip',
                     val_dataset_path='/home/jiahua/singa-local/dataset.zip',
                     train_args={},
                     queries=imgs)
