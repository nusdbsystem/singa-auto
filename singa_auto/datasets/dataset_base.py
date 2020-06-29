import abc
from typing import Tuple, Generic, TypeVar
import PIL
from PIL import Image
import io


def _load_pil_image(image_path, mode='RGB'):
    """
    load one image
    """

    try:
        with open(image_path, 'rb') as f:
            encoded = io.BytesIO(f.read())
            pil_image = Image.open(encoded).convert(mode)
    except:
        print('error accurs when handling : ', image_path)
        raise

    return pil_image


T = TypeVar('T')
K = TypeVar('K')


class ModelDataset(Generic[K, T]):
    '''
    Abstract that helps loading of dataset of a specific type

    ``size`` should be the total number of samples of the dataset
    '''

    @abc.abstractmethod
    def __getitem__(self, index: int) -> Tuple[K, T]:
        raise NotImplementedError()


class DetectionModelDataset(ModelDataset[PIL.Image.Image, dict]):
    pass


class ClfModelDataset(ModelDataset[PIL.Image.Image, int]):
    pass




