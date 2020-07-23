from keras.applications.inception_resnet_v2 import InceptionResNetV2
from .food_objection_base_model import FoodDetectionBase


class FoodDetection256(FoodDetectionBase):

    def __init__(self, **knobs):
        super().__init__(clf_model_class_name=InceptionResNetV2, **knobs)

        # pre config
        self.classes = 256
        self.image_size = 299

        # preload files
        self.yolo_cfg_name = "yolov3-food.cfg"
        self.yolo_weight_name = "yolov3-food_final.weights"
        self.food_name = "food.names"

        # this is the model file downloaded from internet,
        # can choose download locally and upload , or download from server
        # if download at server side, leave it to none
        self.preload_clf_model_weights_name = "inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5"

        # this is the trained model
        self.trained_clf_model_weights_name = "inceptionresnet-U256-0.73.h5"

        self._npy_index_name = "uec256.npy"
