import abc

class BaseExplanation(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_explanation_name(self):
        """
        returns the explanation module's name
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_local_explanation_for_sample(self, model, sample, label):
        """
        returns a local explanation for designated sample 

        parameters:
            model: a pytorch model
            sample: data sample
            label: data label

        return:
            explanation
        """
        raise NotImplementedError()
