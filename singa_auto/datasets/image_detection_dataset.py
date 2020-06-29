import copy
import numpy as np
import os
import zipfile
import tempfile
from singa_auto.datasets.dataset_base import DetectionModelDataset
import torch
import torch.utils.data
from PIL import Image
from pycocotools.coco import COCO
from singa_auto.datasets.torch_utils import get_transform

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class PennFudanDataset(DetectionModelDataset, torch.utils.data.Dataset):
    def __init__(self, dataset_path, is_train):
        self.root_path = None
        self.transforms = get_transform(is_train)

        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs, self.masks = self._extract_zip(dataset_path)

    def __getitem__(self, idx):
        # load images ad masks

        img_path = os.path.join(self.root_path.name, "PennFudanPed", "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root_path.name, "PennFudanPed", "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class in this dataset
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def _extract_zip(self, dataset_path):
        dataset_zipfile = zipfile.ZipFile(dataset_path, 'r')
        self.root_path = tempfile.TemporaryDirectory()
        dataset_zipfile.extractall(path=self.root_path.name)
        imgs = list(sorted(os.listdir(os.path.join(self.root_path.name, "PennFudanPed", "PNGImages"))))
        masks = list(sorted(os.listdir(os.path.join(self.root_path.name, "PennFudanPed", "PedMasks"))))

        return imgs, masks


class CocoDataset(DetectionModelDataset, torch.utils.data.Dataset):
    def __init__(self, dataset_path, annotation_dataset_path, filter_classes, dataset_name, is_train):

        self.dataset_path = dataset_path
        self.transforms = get_transform(is_train)

        self.root_path = None

        year = dataset_name[-4:]
        if is_train:
            self.img_folder_name = "train{}".format(year)
            self.annotation_file_name = "instances_train{}.json".format(year)
        else:
            self.img_folder_name = "val{}".format(year)
            self.annotation_file_name = "instances_val{}.json".format(year)

        self.imgs, self.annotation_file = self._extract_zip(dataset_path, annotation_dataset_path)
        self.coco = COCO(self.annotation_file)
        # eg: filter_classes: ['person', 'dog']
        self.cat_ids = self.coco.getCatIds(catNms=filter_classes)
        self.ids = self.coco.getImgIds(catIds=self.cat_ids)
        # self.ids = self.getCombinedImgIds(catIds=self.cat_ids)
        self.label_mapper = {v: key+1 for key, v in enumerate(self.cat_ids)}

    def __getitem__(self, index):
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image

        img_file_name = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root_path.name, self.img_folder_name, img_file_name))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        masks = list()
        labels = []
        areas = []
        for i in range(num_objs):
            if coco_annotation[i]['category_id'] not in self.cat_ids:
                continue

            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])

            masks.append(coco.annToMask(coco_annotation[i]))

            labels.append(self.label_mapper[coco_annotation[i]["category_id"]])
            areas.append(coco_annotation[i]['area'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)

        # Annotation is in dictionary format
        target = dict()
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = img_id
        target["area"] = areas
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def _extract_zip(self, dataset_path, annotation_dataset_path):
        dataset_zipfile = zipfile.ZipFile(dataset_path, 'r')
        annotation_zipfile = zipfile.ZipFile(annotation_dataset_path, 'r')
        # create temp dir
        self.root_path = tempfile.TemporaryDirectory()
        # extract images and annotations
        dataset_zipfile.extractall(path=self.root_path.name)
        annotation_zipfile.extractall(path=self.root_path.name)
        imgs = list(sorted(os.listdir(os.path.join(self.root_path.name, self.img_folder_name))))
        annotation_file = os.path.join(self.root_path.name, "annotations", self.annotation_file_name)
        return imgs, annotation_file

    def getCombinedImgIds(self, imgIds=[], catIds=[]):
        ids = set(imgIds)

        for i, catId in enumerate(catIds):
            if i == 0 and len(ids) == 0:
                ids = set(self.coco.catToImgs[catId])
            else:
                ids |= set(self.coco.catToImgs[catId])

        return list(ids)
