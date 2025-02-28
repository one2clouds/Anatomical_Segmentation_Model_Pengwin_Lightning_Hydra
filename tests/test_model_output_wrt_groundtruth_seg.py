import unittest
from unittest import TestCase
import torch 
import os 
from monai.networks import nets 
import SimpleITK as sitk 
from monai.transforms import Resized, Compose, LoadImaged, Orientationd, Spacingd, EnsureTyped, EnsureChannelFirstd, AsDiscrete, CastToTyped, Resize, Orientation, Spacing
import numpy as np 
import torch.nn as nn 


val_transform = Compose([
        LoadImaged(keys=["image", "label"]),
        # CastToTyped(keys=["image", "label"], dtype=torch.int8),
        EnsureChannelFirstd(keys=["image","label"]),
        EnsureTyped(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"],pixdim=(1.0, 1.0, 1.0),mode=("bilinear", "nearest"),),
        Resized(keys=["image","label"],spatial_size=(128,128,128), mode=("area", "nearest")),
        # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        # RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ])


class TestModelOutputWithInput(TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.model_path = '/home/shirshak/lightning-hydra-template/logs/train/runs/unet_model_training/checkpoints/best.ckpt'
        self.input_path = '/mnt/Enterprise2/shirshak/PENGWIN_TASK/PENGWIN_CT/part1/001.mha'
        self.label_path = '/mnt/Enterprise2/shirshak/PENGWIN_TASK/PENGWIN_CT/labels/001.mha'

    def test_model_op_wrt_GT(self):
        model = nets.UNet(spatial_dims=3, in_channels=1, out_channels=4, channels=[16,32,64], strides=[2,2])
        model_file_state_dict = torch.load(self.model_path)['state_dict']
        pretrained_dict = {key.replace("net.", ""): value for key, value in model_file_state_dict.items()}
        model.load_state_dict(pretrained_dict)

        data_dicts = [{"image": self.input_path, "label": self.label_path}]
        result = val_transform(data_dicts)

        self.assertEqual(predicted_segmentation.shape, )

        logits = model.forward(result["image"].unsqueeze(0)) # unsqueeze for b, c, h, w, d
        softmax_logits = nn.Softmax(dim=1)(logits)
        predicted_segmentation = torch.argmax(softmax_logits, 1)

        self.assertEqual(predicted_segmentation.shape, )






# python -m unittest tests/test_model_output_wrt_groundtruth_seg.py 

if __name__ == '__main__':
    unittest.main()
    

