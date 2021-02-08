import copy
import cv2
import itertools
import json
import numpy as np
import os
import random
import tempfile
import time
import torch
import torch.utils.data
import zipfile

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from collections import defaultdict
from pycocotools.coco import COCO
from torchvision.transforms import transforms

from singa_auto.darknet.utils import pad_to_square, resize
from singa_auto.datasets.dataset_base import DetectionModelDataset
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


def fectch_from_train_set(root_path, split_ratio=0.8):
    image_train_folder = os.path.join(root_path, "train", "image")
    image_val_folder = os.path.join(root_path, "val", "image")
    annotation_train_folder = os.path.join(root_path, "train", "annotation")
    annotation_val_folder = os.path.join(root_path, "val", "annotation")

    os.makedirs(image_val_folder, exist_ok=True)
    os.makedirs(annotation_val_folder, exist_ok=True)

    list_image = list(sorted(os.listdir(image_train_folder)))
    list_annotation = list(sorted(os.listdir(annotation_train_folder)))

    union_list = []
    for image_name in list_image:
        base_name, _ = os.path.splitext(image_name)

        if base_name + ".json" in list_annotation:
            union_list.append(image_name)

    disordered_index = np.random.permutation(range(len(union_list)))
    val_list = disordered_index[np.int(len(union_list) * split_ratio):]
    import shutil

    for image_idx in val_list:
        image_name = union_list[image_idx]
        annotation_name = os.path.splitext(image_name)[0] + ".json"

        shutil.move(os.path.join(image_train_folder, image_name), os.path.join(image_val_folder, image_name))
        shutil.move(os.path.join(annotation_train_folder, annotation_name), os.path.join(annotation_val_folder, annotation_name))


def split_dataset(root_path, split_ratio=0.8):
    image_path = os.path.join(root_path, "image")
    annotation_path = os.path.join(root_path, "annotation")

    image_train_folder = os.path.join(root_path, "train", "image")
    image_val_folder = os.path.join(root_path, "val", "image")
    annotation_train_folder = os.path.join(root_path, "train", "annotation")
    annotation_val_folder = os.path.join(root_path, "val", "annotation")

    os.makedirs(image_train_folder, exist_ok=True)
    os.makedirs(image_val_folder, exist_ok=True)
    os.makedirs(annotation_train_folder, exist_ok=True)
    os.makedirs(annotation_val_folder, exist_ok=True)

    list_image = list(sorted(os.listdir(image_path)))
    list_annotation = list(sorted(os.listdir(annotation_path)))

    union_list = []
    for image_name in list_image:
        base_name, _ = os.path.splitext(image_name)

        if base_name + ".json" in list_annotation:
            union_list.append(image_name)
    
    disordered_index = np.random.permutation(range(len(union_list)))
    train_list = disordered_index[:np.int(len(union_list) * split_ratio)]
    val_list = disordered_index[np.int(len(union_list) * split_ratio):]

    import shutil
    for image_idx, image_name in enumerate(union_list):
        annotation_name = os.path.splitext(image_name)[0] + ".json"

        if image_idx in train_list:
            shutil.copy(os.path.join(image_path, image_name), os.path.join(image_train_folder, image_name))
            shutil.copy(os.path.join(annotation_path, annotation_name), os.path.join(annotation_train_folder, annotation_name))
        else:
            shutil.copy(os.path.join(image_path, image_name), os.path.join(image_val_folder, image_name))
            shutil.copy(os.path.join(annotation_path, annotation_name), os.path.join(annotation_val_folder, annotation_name))


class YoloCoco(object):
    def __init__(self, annotation_path=None, is_single_json_file=False):
        """
        dataset for YOLO, according with coco
        @ annotation_path: annotation path, filename if a single json, folder path is multiple jsons
        """
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.img_to_ann, self.cat_to_img = defaultdict(list), defaultdict(list)

        if annotation_path is not None:
            print("loading annotations into memory")
            tic = time.time()

            if is_single_json_file:
                # load annotations from single json
                with open(annotation_path, 'r') as f:
                    dataset = json.load(f)
            else:
                # load annotations from json files
                dataset = self.load_scattered_json(annotation_path)

            assert type(dataset)==dict, "annotation file format {} not supported".format(type(dataset))
            print("Done (t={:0.2f}s)".format(time.time()- tic))
            self.dataset = dataset
        else:
            raise ValueError("annotation_path should not be None")

        self.create_index()

    def _is_array_like(self, obj):
        return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

    def load_scattered_json(self, annotation_path):
        """
        merge annotation into a dataset, in accordancy with pycocotool
        """
        list_annotation = list(sorted(os.listdir(annotation_path)))

        dataset = {
            "images": list(),
            "annotations": list(),
            "categories": list(),
        }

        dict_category = dict()
        dict_image = dict()
        last_category_id = 0
        last_annotation_id = 0
        last_image_id = 0

        # for all json files
        for annotation_idx, annotation_filename in enumerate(list_annotation):
            with open(os.path.join(annotation_path, annotation_filename), 'r') as f:
                json_info = json.load(f)
            
            # process image info
            image_id = int(json_info["imagePath"][15:-4])
            if image_id not in dict_image:
                dict_image[image_id] = last_image_id
                last_image_id += 1

                image_info = {
                    "file_name": json_info["imagePath"],
                    "height": json_info["imageHeight"],
                    "width": json_info["imageWidth"],
                    "id": image_id,
                }

                dataset["images"].append(image_info)

            # process bounding box information
            for bounding_box_info in json_info["shapes"]:
                if bounding_box_info["label"] not in dict_category:
                    dict_category[bounding_box_info["label"]] = last_category_id

                    category_info = {
                        "id": last_category_id,
                        "name":bounding_box_info["label"],
                    }

                    dataset["categories"].append(category_info)
                    last_category_id += 1

                annotation_info = {
                    "image_id": image_id,
                    "bbox": list(np.array(np.concatenate((bounding_box_info["points"][0], bounding_box_info["points"][1]), axis=0), dtype=np.int)),
                    "category_id": dict_category[bounding_box_info["label"]],
                    "id": last_annotation_id,
                }
                last_annotation_id += 1

                dataset["annotations"].append(annotation_info)
        return dataset

    def create_index(self):
        print("creating index")
        anns, cats, imgs = dict(), dict(), dict()
        img_to_ann, cat_to_img = defaultdict(list), defaultdict(list)

        if "annotations" in self.dataset:
            for ann in self.dataset["annotations"]:
                img_to_ann[ann["image_id"]].append(ann)
                anns[ann["id"]] = ann

        if "images" in self.dataset:
            for img in self.dataset["images"]:
                imgs[img["id"]] = img

        if "categories" in self.dataset:
            for cat in self.dataset["categories"]:
                cats[cat["id"]] = cat

        if "annotations" in self.dataset and "categories" in self.dataset:
            for ann in self.dataset["annotations"]:
                cat_to_img[ann["category_id"]].append(ann["image_id"])

        print("index created")

        # create class member
        self.anns = anns
        self.cats = cats
        self.imgs = imgs
        self.cat_to_img = cat_to_img
        self.img_to_ann = img_to_ann

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            print('{}: {}'.format(key, value))

    def get_ann_id(self, img_id=[], cat_id=[], area_rng=[], is_crowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param: img_id  (int array)    get anns for given imgs
        :param: cat_id  (int array)    get anns for given cats
        :param: area_rng (float array) get anns for given area range (e.g. [0 inf])
        :param: is_crowd (boolean)     get anns for given crowd label (False or True)
        :return: ids (int array)       integer array of ann ids
        """
        img_id = img_id if self._is_array_like(img_id) else [img_id]
        cat_id = cat_id if self._is_array_like(cat_id) else [cat_id]

        if len(img_id) == len(cat_id) == len(area_rng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(img_id) == 0:
                lists = [self.img_to_ann[imgId] for imgId in img_id if imgId in self.img_to_ann]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(cat_id)  == 0 else [ann for ann in anns if ann['category_id'] in cat_id]
            anns = anns if len(area_rng) == 0 else [ann for ann in anns if ann['area'] > area_rng[0] and ann['area'] < area_rng[1]]
        if not is_crowd is None:
            ids = [ann['id'] for ann in anns if ann['is_crowd'] == is_crowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

    def get_cat_id(self, cat_nms=[], sup_nms=[], cat_id=[]):
        """
        filtering parameters. default skips that filter.
        :param: cat_nms (str array)  : get cats for given cat names
        :param: sup_nms (str array)  : get cats for given supercategory names
        :param: cat_id (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        cat_nms = cat_nms if self._is_array_like(cat_nms) else [cat_nms]
        sup_nms = sup_nms if self._is_array_like(sup_nms) else [sup_nms]
        cat_id = cat_id if self._is_array_like(cat_id) else [cat_id]

        if len(cat_nms) == len(sup_nms) == len(cat_id) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(cat_nms) == 0 else [cat for cat in cats if cat['name']          in cat_nms]
            cats = cats if len(sup_nms) == 0 else [cat for cat in cats if cat['supercategory'] in sup_nms]
            cats = cats if len(cat_id) == 0 else [cat for cat in cats if cat['id']            in cat_id]
        ids = [cat['id'] for cat in cats]
        return ids

    def get_img_id(self, img_id=[], cat_id=[]):
        """
        Get img ids that satisfy given filter conditions.
        :param: img_id (int array)  get imgs for given ids
        :param: cat_id (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        """
        img_id = img_id if self._is_array_like(img_id) else [img_id]
        cat_id = cat_id if self._is_array_like(cat_id) else [cat_id]

        if len(img_id) == len(cat_id) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(img_id)
            for i, cat_id in enumerate(cat_id):
                if i == 0 and len(ids) == 0:
                    ids = set(self.cat_to_img[cat_id])
                else:
                    # original &=, but should be |=
                    ids |= set(self.cat_to_img[cat_id])
        return list(ids)

    def load_ann(self, ids=[]):
        """
        Load anns with the specified ids.
        :param: ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if self._is_array_like(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def load_cat(self, ids=[]):
        """
        Load cats with the specified ids.
        :param: ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if self._is_array_like(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def load_imgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param: ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if self._is_array_like(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]

    def load_numpy_annotation(self, data):
        """
        Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
        :param:  data (numpy.ndarray)
        :return: annotations (python nested list)
        """
        print('Converting ndarray to lists...')
        assert(type(data) == np.ndarray)
        print(data.shape)
        assert(data.shape[1] == 7)
        N = data.shape[0]
        ann = []
        for i in range(N):
            if i % 1000000 == 0:
                print('{}/{}'.format(i,N))
            ann += [{
                'image_id'  : int(data[i, 0]),
                'bbox'  : [ data[i, 1], data[i, 2], data[i, 3], data[i, 4] ],
                'score' : data[i, 5],
                'category_id': int(data[i, 6]),
                }]
        return ann



class YoloDataset(torch.utils.data.Dataset):
    """
    dataset of yolo
    """
    def __init__(self, image_path, annotation_path, is_single_json_file, filter_classes, is_train, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        self.root_path = image_path
        self.imgs = list(sorted(os.listdir(image_path)))
        self.annotation_path = annotation_path
        self.coco = YoloCoco(self.annotation_path, is_single_json_file=is_single_json_file)
        # eg: filter_classes: ['person', 'dog']
        self.cat_ids = self.coco.get_cat_id(cat_nms=filter_classes)
        self.ids = self.coco.get_img_id(cat_id=self.cat_ids)
        
        self.cat_to_label = {v: key+1 for key, v in enumerate(self.cat_ids)}
        self.label_to_cat = {key+1: v for key, v in enumerate(self.cat_ids)}

        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment if is_train else False
        self.multiscale = multiscale if is_train else False
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

        # if os.path.exists(r"./rectangle_images/"):
        #     import shutil
        #     shutil.rmtree(r"./rectangle_images/")
        # os.makedirs(r"./rectangle_images/", exist_ok=True)

    def __getitem__(self, index):
        img_id = self.ids[index % len(self.ids)]
        ann_id = self.coco.get_ann_id(img_id=img_id)

        img_path = os.path.join(self.root_path, self.coco.load_imgs(img_id)[0]["file_name"])

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
        coco_annotation = self.coco.load_ann(ann_id)

        tmp_label = []
        box_info = []
        for ann in coco_annotation:
            if ann["category_id"] not in self.cat_ids:
                continue
            boxes = torch.zeros((1, 6), dtype=torch.float32)
            x1 = round(max(ann['bbox'][0], 0))
            y1 = round(max(ann['bbox'][1], 0))
            x2 = round(min(x1 + ann['bbox'][2], w - 1))
            y2 = round(min(y1 + ann['bbox'][3], h - 1))

            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]

            box_info.append(((x1, y1), (x2, y2)))

            # print(x1, x2, y1, y2, padded_h, padded_w)
            # Returns (x, y, w, h)
            boxes[0, 2] = (x2 + x1) / 2 / padded_w
            boxes[0, 3] = (y2 + y1) / 2 / padded_h
            boxes[0, 4] = (x2 - x1) / padded_w
            boxes[0, 5] = (y2 - y1) / padded_h
            boxes[0, 1] = self.cat_to_label[ann["category_id"]]
            tmp_label.append(boxes)
        
        # self.get_bounding_box(img, os.path.basename(img_path), box_info)

        # targets from list to tensor
        targets = torch.cat(tmp_label, dim=0)

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = self.horisontal_flip(img, targets)

        return img_path, img, targets

    def __len__(self):
        return len(self.ids)

    def _extract_zip(self, dataset_path, annotation_path):
        dataset_zipfile = zipfile.ZipFile(dataset_path, 'r')
        annotation_zipfile = zipfile.ZipFile(annotation_path, 'r')

        # create temp dir
        self.root_path = tempfile.TemporaryDirectory()

        # extract images and annotations
        dataset_zipfile.extractall(path=self.root_path.name)
        annotation_zipfile.extractall(path=self.root_path.name)
        imgs = list(sorted(os.listdir(os.path.join(self.root_path.name, self.img_folder_name))))
        annotation_file = os.path.join(self.root_path.name, "annotations", self.annotation_file_name)

        return imgs, annotation_file

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

    # def get_bounding_box(self, img, basename, boxes, rect_th=3):
    #     """
    #     draw the bounding box on img
    #     """
    #     tmp = img.squeeze().detach().permute((1, 2, 0)).mul(255).clamp(0, 255).numpy()
    #     tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR)
    # 
    #     for rect_info in boxes:
    #         cv2.rectangle(tmp, rect_info[0], rect_info[1], (0, 255, 0), rect_th)
    # 
    #     cv2.imwrite('./rectangle_images/{}'.format(basename), tmp)

    def horisontal_flip(self, images, targets):
        images = torch.flip(images, [-1])
        targets[:, 2] = 1 - targets[:, 2]
        return images, targets


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
