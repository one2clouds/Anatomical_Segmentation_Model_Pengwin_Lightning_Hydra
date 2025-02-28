from typing import Any, Dict, Optional, Tuple
import torch
from lightning import LightningDataModule
from monai.transforms import (Resized,EnsureType,Activations,Activationsd,AsDiscrete,AsDiscreted,Compose,Invertd,LoadImaged,NormalizeIntensityd,Orientationd,RandFlipd,RandScaleIntensityd,RandSpatialCropd,Spacingd,EnsureTyped,EnsureChannelFirstd,RandShiftIntensityd)
from torch.utils.data import DataLoader, Dataset
import glob
import os
from monai.utils import set_determinism

# the resize doesnot use bilinear for image, because idk it gets 5d data and bilinear uses 4d hence we could use either area or trilinear as trilinear uses 5d data
train_transform = Compose([
            # load 4 Nifti images and stack them together
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image","label"]),
            EnsureTyped(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"],pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest"),),
            Resized(keys=["image","label"],spatial_size=(128,128,128), mode=("area", "nearest")),
            # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            # RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            # RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ])

# the resize doesnot use bilinear for image, because idk it gets 5d data and bilinear uses 4d hence we could use either area or trilinear as trilinear uses 5d data
val_transform = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image","label"]),
            EnsureTyped(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"],pixdim=(1.0, 1.0, 1.0),mode=("bilinear", "nearest"),),
            Resized(keys=["image","label"],spatial_size=(128,128,128), mode=("area", "nearest")),
            # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            # RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            # RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ])

def  make_discrete_data(data):
    # print(data.unique())
    discreted_data = torch.zeros_like(data)
    original_data = torch.clone(data)

    discreted_data[(data>0) & (data<=10)] = 1
    discreted_data[(data>10) & (data<=20)] = 2
    discreted_data[(data>20) & (data<=30)] = 3

    return discreted_data, original_data 

class HelperDataset(Dataset):
    def __init__(self, file_names, transform):
        self.file_names = file_names
        self.transform = transform

    def __getitem__(self, index):
        file_names = self.file_names[index]
        dataset = self.transform(file_names) 
        # print(dataset["image"].shape)
        # print(dataset["label"].unique())
        dataset["label"], _ = make_discrete_data(dataset["label"])
        return dataset
    
    def __len__(self):
        return len(self.file_names)

root_dir = '/mnt/Enterprise2/shirshak/PENGWIN_TASK/PENGWIN_CT/'

class PelvicFracModule(LightningDataModule):
    def __init__(self, batch_size: int = 8, num_workers: int=0, pin_memory: bool=False, num_classes:int=5) -> None:
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.batch_size_per_device = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        # Divide batch size by the number of devices.

        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            train_images = sorted(glob.glob(os.path.join(root_dir, "p*", "*.mha")))
            train_labels = sorted(glob.glob(os.path.join(root_dir, "la*", "*.mha")))
            data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
            # print(data_dicts[55])
            set_determinism(seed=12345)
            train_files, val_files, test_files = data_dicts[:-20], data_dicts[-20:-10], data_dicts[-10:]
            self.data_train = HelperDataset(train_files, train_transform)
            self.data_val = HelperDataset(val_files, val_transform)
            self.data_test = HelperDataset(test_files, val_transform)

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,)

if __name__ == "__main__":
    _ = PelvicFracModule()
