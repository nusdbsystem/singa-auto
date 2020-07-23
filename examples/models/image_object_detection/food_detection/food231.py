from keras.applications.xception import Xception
from .food_objection_base_model import FoodDetectionBase


class FoodDetection231(FoodDetectionBase):

    def __init__(self, **knobs):
        super().__init__(clf_model_class_name=Xception, **knobs)

        # pre config
        self.classes = 231
        self.image_size = 299

        # preload files
        self.yolo_cfg_name = "yolov3-food.cfg"
        self.yolo_weight_name = "yolov3-food_final.weights"
        self.food_name = "food.names"

        # this is the model file downloaded from internet,
        # can choose download locally and upload , or download from server
        # if download at server side, leave it to none
        self.preload_clf_model_weights_name = None

        # this is the trained model
        self.trained_clf_model_weights_name = "xception-food231-0-15-0.82.h5"

        self._npy_index_name = "food231.npy"
