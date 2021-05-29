from abc import ABC, abstractmethod


class ModelMaker(ABC):
    @abstractmethod
    def create(self):
        pass

    @abstractmethod
    def save(self):
        pass