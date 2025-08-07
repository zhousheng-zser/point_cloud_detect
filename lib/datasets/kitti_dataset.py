import os
import numpy as np
import torch.utils.data as torch_data
import lib.utils.calibration as calibration
import lib.utils.kitti_utils as kitti_utils
from PIL import Image


class KittiDataset(torch_data.Dataset):
    def __init__(self, root_dir, split='train'):
        self.split = split
        is_test = self.split == 'test'
        #self.imageset_dir = os.path.join(root_dir, 'KITTI', 'object', 'testing' if is_test else 'training')

        #split_dir = os.path.join(root_dir, 'KITTI', 'ImageSets', split + '.txt')
        #self.image_idx_list = [x.strip() for x in open(split_dir).readlines()]
        #self.num_sample = self.image_idx_list.__len__()

        #self.image_dir = os.path.join(self.imageset_dir, 'image_2')
        #self.lidar_dir = os.path.join(self.imageset_dir, 'velodyne')
        self.calib_dir = root_dir
        #self.label_dir = os.path.join(self.imageset_dir, 'label_2')
        #self.plane_dir = os.path.join(self.imageset_dir, 'planes')

    def get_calib(self):
        calib_file = os.path.join(self.calib_dir, 'calib.txt')
        assert os.path.exists(calib_file)
        return calibration.Calibration(calib_file)


    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError
