import abc

class BaseExplanation(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_explanation_name(self):
        raise NotImplementedError


