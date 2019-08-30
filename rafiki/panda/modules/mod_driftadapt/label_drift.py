import json

import torch
import torch.nn

import numpy as np

from rafiki.panda.modules.mod import BaseMod


class LabelDriftAdapter(BaseMod):
    def __init__(self, model, num_classes):
        self.model = model
        self.num_classes = num_classes
        self.C = np.zeros((self.num_classes, self.num_classes))
        self.Cinv = np.zeros((self.num_classes, self.num_classes))
        self.count_val = 0
    
    def get_mod_name(self):
        return "mod_label_drift_adapter"

    def dump_parameters(self) -> str:
        params = {}
        params["cinv"] = self.Cinv.tolist()
        return json.dumps(params)
    
    def load_parameters(self, params:str):
        params_dict = json.loads(params)
        self.Cinv = np.array(params_dict["cinv"])

    def accumulate_c(self, outputs, targets):
        """
            accumulate the confusion matrix, should be called after each epoch
        """
        _, predicted = outputs.max(1)
        predicted = predicted.cpu().data.numpy()
        targets = targets.cpu().data.numpy()

        for i in range(0, predicted.shape[0]):
            self.C[predicted[i]][targets[i]] += 1
        
        self.count_val += predicted.shape[0]

    def estimate_cinv(self):
        """
        calculate Cinv matrix base on accumulated confusion matrix
        """
        C = self.C * (1.0/self.count_val)
        Cinv = np.linalg.inv(C)
        self.Cinv = Cinv
        print(C)
        print(Cinv)

    def adapt(self, outputs):
        """
        perform label adapt with cinv estimated

        params:
            outputs:

        return:
        """
        batch_size = outputs.shape[0]

        _, predicted = outputs.max(1)
        predicted = predicted.cpu().tolist()
        miu_est = np.zeros((self.num_classes,))

        for i in range(0, len(predicted)):
            miu_est[predicted[i]] += 1
        print('miu_est: ', miu_est*(1/batch_size))

        w_est = np.matmul(self.Cinv, miu_est*(1/batch_size))
        w_est = torch.from_numpy(w_est).cuda()
        print('w_est: ', w_est)

        softmax = torch.nn.Softmax(dim=1)

        outputs_prob = softmax(outputs)
        outputs_prob_adapt = torch.mul(outputs_prob.double(), w_est.double())
        _, predicted_adapt = outputs_prob_adapt.max(1)

        return outputs_prob_adapt
