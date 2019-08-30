"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import cv2
import numpy as np
import torch

from extractor import CamExtractorAlexNet, CamExtractorDenseNet, CamExtractorResNet, CamExtractorVGG

class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, model_arch, target_layer):
        self.model = model.model_ft
        self.model.eval()
        self.model_arch = model_arch
        # Define extractor
        if model_arch == 'densenet':
            self.extractor = CamExtractorDenseNet(self.model, target_layer)
        elif model_arch == 'alexnet':
            self.extractor = CamExtractorAlexNet(self.model, target_layer)
        elif model_arch == 'resnet':
            self.extractor = CamExtractorResNet(self.model, target_layer)
        elif model_arch == 'vgg':
            self.extractor = CamExtractorVGG(self.model, target_layer)
        else:
            raise Exception()

    def generate_cam(self, input_image, target_class=None):
        """
        generate a grad cam saliency map
        
        params:
            input_image: np.ndarray 
                size of (1, 3, W, H), the image should be normalized before being fed into this routine
            target_class: int 
                the number of class

        return:
            cam: np.ndarray 
                generated saliency map size of (1, W, H)
        """
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        input_image = torch.FloatTensor(input_image)
        conv_output, model_output = self.extractor.forward_pass(input_image)

        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        if self.model_arch == "densenet":
            self.model.features.zero_grad()
            self.model.classifier.zero_grad()
        elif self.model_arch == "alexnet":
            self.model.features.zero_grad()
            self.model.classifier.zero_grad()
        elif self.model_arch == "resnet":
            self.model.conv1.zero_grad()
            self.model.bn1.zero_grad()
            self.model.relu.zero_grad()
            self.model.maxpool.zero_grad()

            self.model.layer1.zero_grad()
            self.model.layer2.zero_grad()
            self.model.layer3.zero_grad()
            self.model.layer4.zero_grad()
        elif self.model_arch == "vgg":
            self.model.features.zero_grad()
            self.model.classifier.zero_grad()
        else:
            raise Exception()

        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        
        # Generate readable image
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cv2.resize(cam, dsize=(input_image.shape[2], input_image.shape[3]), interpolation=cv2.INTER_CUBIC)
        return cam

if __name__ == '__main__':
    pass

