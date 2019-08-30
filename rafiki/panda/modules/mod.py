import abc

class BaseMod(abc.ABC):
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def get_mod_name(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def dump_parameters(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def load_parameters(self, params: str):
        raise NotImplementedError