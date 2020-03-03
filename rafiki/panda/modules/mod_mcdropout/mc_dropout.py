import torch.nn.functional as F
import torch.nn as nn

class MCDropout(nn.Dropout):
    def __init__(self, p=0.5, inplace=False):
        super(MCDropout, self).__init__(p=p, inplace=inplace)
    
    def forward(self, input):
        return F.dropout(input, p=self.p, training=True, inplace=self.inplace)

"""
def MCDropout(in_vec, p=0.5, mask=True):
    return F.dropout(in_vec, p=p, training=mask, inplace=True)
"""

def update_model(torch_model):
    def update(model):
        for name, module in model.named_children():
            if isinstance(module, nn.Dropout):
                print(module)
                setattr(model, name, MCDropout(module.p, module.inplace))
            update(module)

    update(torch_model)

    return torch_model

