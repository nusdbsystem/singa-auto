from lime import lime_image

import torchvision.transforms as transforms
from skimage.segmentation import mark_boundaries
from skimage import io
import numpy as np
import torch
import torch.nn.functional as F
from rafiki.model import utils


class Lime():
    """
    Lime: Explaining the predictions of any machine learning classifier
    https://github.com/marcotcr/lime
    """

    def __init__(self, model, image_size, normalize_mean, normalize_std, use_gpu):
        self._model = model
        self._use_gpu = use_gpu
        # dataset
        self._image_size = image_size
        self._normalize_mean = normalize_mean
        self._normalize_std = normalize_std
        self._explainer = lime_image.LimeImageExplainer()
        # lime configs
        # number of images that will be sent to classification function
        self._num_samples = 100
        self._top_labels = 5
        self._hide_color = 0

    def batch_predict(self, images):
        (images, _, _) = utils.dataset.normalize_images(images, self._normalize_mean, self._normalize_std)

        self._model.eval()

        # images are size of (B, W, H, C)
        with torch.no_grad():
            images = torch.FloatTensor(images).permute(0, 3, 1, 2)
        logits = self._model(images)
        probs = F.softmax(logits, dim=1)

        return probs.detach().cpu().numpy()

    def explain(self, images):
        img_boundry = []
        for img in images:
            explanation = self._explainer.explain_instance(img, self.batch_predict, self._top_labels, self._hide_color,
                                                           self._num_samples)
            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                                        hide_rest=False)
            # (M, N, 3) array of float
            img_boundry = mark_boundaries(temp / 255.0, mask)
        return img_boundry*255


'''
if method == 'lime':
    self._lime = Lime(self._model, self._image_size, self._normalize_mean, self._normalize_std, self._use_gpu)
    results = self._lime.explain(queries)
    return results
'''

