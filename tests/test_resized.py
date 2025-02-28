from monai.transforms import Spacingd, Resized, Orientationd, LoadImaged, Compose, EnsureChannelFirstd, Spacing, LoadImage
from unittest import TestCase
import numpy as np
import unittest
import SimpleITK as sitk

class TestTransformInverse(TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.input_img_name = '/mnt/Enterprise2/shirshak/PENGWIN_TASK/PENGWIN_CT/part1/001.mha'

        self.input_img = sitk.ReadImage('/mnt/Enterprise2/shirshak/PENGWIN_TASK/PENGWIN_CT/part1/001.mha')
        self.label_img = sitk.ReadImage('/mnt/Enterprise2/shirshak/PENGWIN_TASK/PENGWIN_CT/labels/001.mha')

    def test_spacing_inv(self):
        required_spacing = (1.0, 1.0, 1.0)
        input_arr = LoadImage(image_only=True)(self.input_img_name)
        spacer = Compose([
            LoadImaged(keys="image"),
            Spacingd(keys="image", pixdim=required_spacing, mode="bilinear")
        ])

        output_arr = spacer({"image":self.input_img_name})

        print(input_arr.shape)
        print(output_arr["image"].shape)

        print(output_arr)

        print(output_arr["image_meta_dict"]["pixdim[1]"])        
        print(output_arr["image_meta_dict"]["pixdim[2]"])
        print(output_arr["image_meta_dict"]["pixdim[3]"])




        # out_spacing = output_arr["image_meta_dict"]["spacing"]
        # self.assertListEqual(list(spacing), list(out_spacing), f"assertion error expected spacing {spacing} but got spacing {out_spacing}")

        spaced_output_arr = spacer.inverse(output_arr)
        print(spaced_output_arr["image_meta_dict"]["pixdim"])
        print(self.input_img.GetSpacing())


        print(f"But the size of the output resized array is : {spaced_output_arr['image'].shape}")

        reverted_spacing = spaced_output_arr["image_meta_dict"]["spacing"]

        self.assertListEqual(list(self.input_img.GetSpacing()), list(reverted_spacing), f"Resized Inverse Failed, expected {self.input_img.GetSpacing()} got {reverted_spacing}")

        # print("Spacing Completed Successfully")

        # # Now Loading from SITK 
        # # spacer = Spacing(pixdim=(1.0,1.0,1.0), mode="bilinear")
        # # output_array = spacer(sitk.GetArrayFromImage(self.input_img))

        # # print(output_array.shape)
        # # print(sitk.GetArrayFromImage(self.input_img).shape)


    def test_orientation_inv(self):
        pass 
        # orientation = Compose(
        #     LoadImaged(keys="image"),
        #     Orientationd(keys="image", axcodes='RAS')
        #     )
        # output_arr = orientation(self.input_img_name)
        # print(output_arr[1]["spacing"])
        # print(self.input_img.GetDirection())

        # output_inverse_orientation = orientation.inverse(output_arr)

        # print()

    def test_combination_of_transforms(self):
        pass
        # combination = Compose([
        #     LoadImage(),
        #     EnsureChannelFirst(),
        #     Orientation(),
        #     Spacing(),
        #     Resize()
        # ])
    
    def test_resize_inv(self):
        pass
        # resize to self.resized_to and then apply inverse() and see if it resizes to self.orig_size
        # size = (128,128,128)
        # resizer = Compose([
        #     LoadImage(),
        #     Resize(size)
        # ])
        # output_arr = resizer(self.input_img_name)

        # print(output_arr[0].shape)
        # print(size)
        

        # # because resizer takes channel also so we take oth value
        # self.assertEqual(output_arr[0].shape,size,f"Resize failed. expected {size} got {output_arr[0].size()} ")
        # resized_back_to_original = resizer.inverse(output_arr)

        # print(resized_back_to_original.shape)
        # print(sitk.GetArrayFromImage(self.input_img).shape)

        # self.assertEqual(resized_back_to_original.shape, sitk.GetArrayFromImage(self.input_img).shape, f"Resized Inverse Failed, expected {sitk.GetArrayFromImage(self.input_img).shape} got {resized_back_to_original.shape}")

        # print("Resize Completed Successfully ")



# python -m unittest tests/test_resized.py


if __name__ == '__main__':
    unittest.main()