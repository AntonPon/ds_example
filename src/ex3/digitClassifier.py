from __future__ import annotations

import torch
import numpy as np
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier

from network import BasicNet

RANDOM_SEED = 43
np.random.seed(RANDOM_SEED)


class DigitClassificationInterface(ABC):

    @abstractmethod
    def train(self, num_epochs: int, data_loader: torch.utils.data.Dataset, device: str = 'cpu') -> tuple[list, list]:
        pass

    @abstractmethod
    def predict(self, img: np.ndarray,  device: str = 'cpu') -> int:
        pass

    @abstractmethod
    def get_info(self):
        pass

    @abstractmethod
    def save_model(self, path: str = './') -> str:
        pass

    @abstractmethod
    def load_model(self, path: str = './') -> str:
        pass


class ConvNetClassifier(DigitClassificationInterface):

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss: torch.nn.Module,
                 input_shape: list[int], summary_writer=None):
        self.model = model
        self.optim = optimizer
        self.loss = loss
        self.input_shape = input_shape
        if summary_writer is not None:
            self.summary_writer = summary_writer

    def train(self, num_epochs: int, data_loader: torch.utils.data.Dataset, cuda: str = 'cpu') -> tuple[list, list]:
        raise NotImplementedError

    def predict(self, img: np.ndarray ,  device: str = 'cpu') -> int:
        img_shape = img.shape

        if len(img_shape) != 3 or list(img_shape) != self.input_shape:
            raise ValueError('Invalid input data shape')
        with torch.no_grad():
            self.model.eval()
            data = torch.from_numpy(img).to(device, torch.float)
            pred = self.model(data)
        return torch.argmax(pred).item()

    def get_info(self):
        print(self.model)

    def save_model(self, path: str = './') -> str:
        raise NotImplementedError

    def load_model(self, path: str = './') -> str:
        raise NotImplementedError


class RandomForestClassifierModel(DigitClassificationInterface):

    def __init__(self, model: RandomForestClassifier, input_shape: list[int]):
        self.model = model
        self.input_shape = input_shape

    def train(self, num_epochs: int, data_loader: torch.utils.data.Dataset, cuda: str = 'cpu') -> tuple[list, list]:
        raise NotImplementedError

    def predict(self, img: np.ndarray,  device: str = 'cpu') -> int:
        img_shape = img.shape
        if len(img_shape) != 3 or list(img_shape) != self.input_shape:
            raise ValueError('Invalid input data shape')
        img = img.reshape(1, -1)
        pred  = self.model.predict(img)
        return pred[0]

    def get_info(self):
        print(self.model)

    def save_model(self, path: str = './') -> str:
        raise NotImplementedError

    def load_model(self, path: str = './') -> str:
        raise NotImplementedError


class RandomModelClassifier(DigitClassificationInterface):

    def __init__(self, input_shape: list[int], total_clas: int = 10):
        self.input_shape = input_shape
        self.total_clas = total_clas

    def train(self, num_epochs: int, data_loader: torch.utils.data.Dataset, cuda: str = 'cpu') -> tuple[list, list]:
        raise NotImplementedError

    def predict(self, img: np.ndarray ,  device: str = 'cpu') -> int:
        img_shape = img.shape
        if len(img_shape) != 2 or list(img_shape) != self.input_shape:
            raise ValueError('Invalid input data shape')

        return np.random.randint(1, self.total_clas)

    def get_info(self):
        print('random model with a seed {}'.format(RANDOM_SEED))

    def save_model(self, path: str = './') -> str:
        raise NotImplementedError

    def load_model(self, path: str = './') -> str:
        raise NotImplementedError


class DigitClassifier:

    def __init__(self, model_type: str,  input_shape: list[int],  model: torch.nn.Module | RandomForestClassifier | None = None,
                 optimizer: torch.optim.Optimizer | None = None, loss: torch.nn.Module | None = None, summary_writer=None):
        if model_type == 'cnn':
            self.model = ConvNetClassifier(model, optimizer, loss, input_shape, summary_writer)
        elif model_type == 'rf':
            self.model = RandomForestClassifierModel(model, input_shape)
        elif model_type == 'rand':
            self.model = RandomModelClassifier(input_shape)
        else:
            raise ValueError('such model does not exist')
        self.model_type = model_type

    def predict(self, img: np.ndarray):
        return self.model.predict(img)


if __name__ == '__main__':
    network = BasicNet()
    optim = torch.optim.Adam(network.parameters(), 0.003)
    loss = torch.nn.NLLLoss()
    input_shape = [1, 28, 28]
    img = np.random.uniform(size=input_shape)
    digit_cl1 = DigitClassifier('cnn', input_shape, network, optim, loss)
    print('cnn predict:', digit_cl1.predict(img))

    forest = RandomForestClassifier(10)
    forest.fit(img.reshape(1, -1), np.array([5]))
    digit_cl2 = DigitClassifier('rf', input_shape, forest)
    print('forest predict:', digit_cl2.predict(img))

    centre_crop = np.random.uniform(size=[10, 10])
    digit_random = DigitClassifier('rand', [10, 10])
    print('randon predict:', digit_random.predict(centre_crop))




