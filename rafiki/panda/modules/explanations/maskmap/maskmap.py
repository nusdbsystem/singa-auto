import cv2
import numpy as np
import torch

class RandomMask(object):
    """
    encapsulation of random mask based saliency map generation
    """
    def __init__(self, model, kernel, stride, mask_value, use_gpu=True):
        self.model = model
        self.use_gpu = use_gpu
        if use_gpu:
            self.model.cuda()
        self.model.eval()

        self.kernel_x, self.kernel_y = kernel
        self.stride_x, self.stride_y = stride
        self.mask_value = mask_value

    def get_occlude_data(self, inputs, height_idx, width_idx):
        """
        mask designated area with designated value

        params:
            inputs:
                input images, size of (batch_size, 3, W, H)
            height_idx:
            width_idx:

        return:
            occluded images

        """
        mask_x0 = width_idx * self.stride_x
        mask_y0 = height_idx * self.stride_y
        mask_x1 = mask_x0 + self.kernel_x
        mask_y1 = mask_y0 + self.kernel_y

        inputs[:, :, mask_x0:mask_x1, mask_y0:mask_y1] = float(255.0)

        return inputs

    def generate_random_mask_saliency_map(self, inputs, label_indices, model):
        """
        generate the random mask based saliency map for a batch of inputs

        params:
            inputs:
                np.ndarray of size (batch_size, 3, W, H)
            label_indices:
                np.ndarray of size (batch_size, )
                the target class for each sample in the batch
            model:
                pretrained pytorch model

            the mask moves at step stride_x, stride_y and is sized of kernel_x, kernel_y

        return:
            saliency map:
                np.ndarray of size (batch_size, W, H)

        """
        inputs = np.array(inputs).astype(np.float32)
        batch_size, channel_n, width, height = inputs.shape

        height_dim = (height - self.kernel_y) / self.stride_y + 1
        width_dim = (width - self.kernel_x) / self.stride_x + 1

        label_indices = np.array(label_indices).astype(np.long)

        probs = np.zeros((int(batch_size), int(width_dim) * int(height_dim)))
        loc = 0

        for width_idx in range(int(width_dim)):
            for height_idx in range(int(height_dim)):
                x = self.get_occlude_data(inputs.copy(), height_idx, width_idx)
                x = torch.FloatTensor(torch.from_numpy(x))

                if self.use_gpu:
                    x = x.cuda()

                out = model(x)
                out = torch.sigmoid(out).data.cpu().numpy()

                n = 0
                for row in out:
                    probs[n, loc] = row[label_indices[n]]
                    n += 1
                loc = loc + 1

        rowmax = np.max(probs, axis=1)
        probs = probs / rowmax[:, None]
        probs = 1 - probs
        probs = probs.reshape((int(batch_size), int(width_dim), int(height_dim)))

        out = np.zeros((batch_size, width, height))
        n = 0
        for prob_image in probs:
            out[n] = cv2.resize(prob_image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
            n += 1

        return out

if __name__ == '__main__':
    pass
