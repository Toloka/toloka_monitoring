import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor

transforms = Compose([
    Resize((224, 224)),
    ToTensor(),
])

class CatsVsDogsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_path,
        batch_size=32,
        num_workers=4,
        train_transform=None,
        test_transform=None,
        **kwargs,
    ):
        super().__init__()
        self.root_path = root_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.test_transform = test_transform

        self.train = None
        self.val = None

    def setup(self, stage=None):
        train_dataset = ImageFolder(self.root_path + '/train', transform=self.train_transform)
        val_dataset = ImageFolder(self.root_path + '/valid', transform=self.test_transform)

        self.train = train_dataset
        self.val = val_dataset

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)
