from lime import lime_image
import torchvision.transforms as transforms
from skimage.segmentation import mark_boundaries
from skimage import io
import numpy as np
import torch
from rafiki.model import utils
class Lime():
    """
    Explain machine learning classifiers using lime
    """
    def __init__(self, model):
        self._model = model
        self._num_samples = 10
        self._explainer = lime_image.LimeImageExplainer()
        
    def get_preprocess_transform(self):
        normalize = transforms.Normalize(self._normalize_mean,
                                        self._normalize_std)     
        preprocess_transform = transforms.Compose([transforms.ToTensor(),normalize])    
        return preprocess_transform
    def get_pil_transform(self): 
        transf = transforms.Compose([transforms.Resize((128, 128))])    
        return transf
    def batch_predict(self, images):
        images = utils.dataset.transform_images(images, image_size=128, mode='RGB')
        (batch, _, _) = utils.dataset.normalize_images(images, self._normalize_mean, self._normalize_std)

        #batch = torch.stack(tuple(self._preprocess_transform(i) for i in images), dim=0)
        self._model.eval()
        with torch.no_grad():
            try:
                images = torch.FloatTensor(batch).permute(0, 3, 1, 2).cuda()
            except Exception:
                images = torch.FloatTensor(batch).permute(0, 3, 1, 2)
        with torch.no_grad():
            outs = self._model(images)
        outs = torch.sigmoid(outs).cpu().numpy()

        return outs

    def explain(self, quires, normalize_mean, normalize_std):
        print("Lime Start\n")
        self._normalize_mean = normalize_mean
        self._normalize_std = normalize_std
        self._preprocess_transform = self.get_preprocess_transform()
        self._transf=self.get_pil_transform()
        index = 0
        for img in quires:
            top_labels = 5
            hide_color = 0
            print(type(img))
            #explanation = self._explainer.explain_instance(np.array(self._transf(img)), self.batch_predict, top_labels, hide_color, self._num_samples)
            explanation = self._explainer.explain_instance(np.array(utils.dataset.transform_images(img, image_size=128, mode='RGB')[0]), self.batch_predict, top_labels, hide_color, self._num_samples)
            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
            img_boundry = mark_boundaries(temp/255.0, mask)
            # integrate with front end later
            io.imsave(str(index) +".png",img)
            io.imsave(str(index) +"lime_top5.png",img_boundry)
            index += 1

