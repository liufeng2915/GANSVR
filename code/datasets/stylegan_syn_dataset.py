import os
import torch
import numpy as np
import cv2
from PIL import Image
import glob
from torchvision import transforms

class SynViewDataset(torch.utils.data.Dataset):
    """load pseudo multi-view images."""

    def __init__(self, data_dir, img_res):

        self.instance_dir = data_dir
        self.instance_list = sorted(os.listdir(self.instance_dir))
        self.n_instances = len(self.instance_list)
        assert os.path.exists(self.instance_dir), "Data directory is empty!"

        self.total_pixels = img_res * img_res
        self.img_res = img_res

        self.trans_rgb = transforms.Compose([
            transforms.Resize(img_res),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
            ])
        self.trans_img = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self.trans_input_img = transforms.Compose([
            transforms.Resize(150),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return self.n_instances

    def load_rgb(self, img_list):

        img_data_list = [Image.open(temp_img_list) for temp_img_list in img_list]

        img = torch.stack([self.trans_img(temp_img_data_list) for temp_img_data_list in img_data_list])
        input_img = self.trans_input_img(img_data_list[0])
        img[0] = input_img

        rgb = torch.stack([self.trans_rgb(temp_img_data_list) for temp_img_data_list in img_data_list])
        rgb = rgb[1:]
        rgb = rgb.view(rgb.shape[0], rgb.shape[1], -1).permute(0,2,1)

        return img, rgb

    def load_mask(self, mask_list):

        alpha = np.array([cv2.resize(cv2.imread(temp_mask_list, cv2.IMREAD_GRAYSCALE),(self.img_res, self.img_res)) for temp_mask_list in mask_list])
        alpha = alpha.astype(np.float32)
        object_mask = alpha > 127.5

        object_mask = object_mask.reshape(object_mask.shape[0], -1)
        object_mask = object_mask[1:]
        object_mask = torch.from_numpy(object_mask).bool()

        return object_mask

    def __getitem__(self, idx):

        # rgb
        instance_img_path = os.path.join(self.instance_dir, self.instance_list[idx], 'image')
        img_list = sorted(glob.glob(instance_img_path+'/*.png'))
        img, rgb = self.load_rgb(img_list)

        # mask
        instance_mask_path = os.path.join(self.instance_dir, self.instance_list[idx], 'mask')
        mask_list = sorted(glob.glob(instance_mask_path+'/*.png'))
        object_mask = self.load_mask(mask_list)

        #
        sample = {
            "image": img, 
            "object_mask": object_mask,
        }
        ground_truth = {
            "rgb": rgb
        }

        return idx, sample, ground_truth