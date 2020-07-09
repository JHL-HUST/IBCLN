"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torchvision.transforms as transforms

import numpy as np

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['I'], sample['T']

        h, w = image.shape[:2]
        min_a = min(h, w)
        self.output_size = (min_a * 7 // 10, min_a * 7 // 10)
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks[top: top + new_h,
                      left: left + new_w]

        return {'I': image, 'T': landmarks}


class ResizeNatural3Dataset(BaseDataset):
    """A reflection dataset class to load data from A1, A2, B datasets, where A1(indoor) and A2(outdoor) are image sets
     without reflection, and B is a image set with reflection."""

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        if opt.phase == 'train':
            self.natural_dir_A1 = os.path.join(opt.dataroot, 'natural_' + 'T')
            self.natural_dir_B = os.path.join(opt.dataroot, 'natural_' + 'I')

            self.natural_A1_paths = sorted(make_dataset(self.natural_dir_A1, opt.max_dataset_size))  # load images from '/path/to/data/trainA1'
            self.natural_B_paths = sorted(make_dataset(self.natural_dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
            self.natural_size = len(self.natural_A1_paths)  # get the size of dataset A1

        self.crop = RandomCrop(opt.load_size)
        self.dir_A1 = os.path.join(opt.dataroot, opt.phase + 'A1')
        self.dir_A2 = os.path.join(opt.dataroot, opt.phase + 'A2')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A1_paths = sorted(make_dataset(self.dir_A1, opt.max_dataset_size))  # load images from '/path/to/data/trainA1'
        self.A2_paths = sorted(make_dataset(self.dir_A2, opt.max_dataset_size)) # load images from '/path/to/data/trainA2'
        if not opt.phase == 'train':
            self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
            self.B_size = len(self.B_paths)  # get the size of dataset B


        self.A1_size = len(self.A1_paths)  # get the size of dataset A1
        self.A2_size = len(self.A2_paths)  # get the size of dataset A2

        input_nc =self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.output_nc  # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        print(self.transform_A)

        self.trans2 = transforms.Compose([transforms.Resize([128, 128]), transforms.ToTensor()])
        self.trans4 = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        """
        is_natural = random.random() <= 0.3
        if self.opt.phase == 'train':
            if is_natural:
                natural_index = index % self.natural_size
                A1_path = self.natural_A1_paths[natural_index]  # make sure index is within then range
                B_path = self.natural_B_paths[natural_index]

                A1_img = np.asarray(Image.open(A1_path).convert('RGB'))
                A2_img = Image.fromarray(np.zeros_like(A1_img))
                B_img = np.asarray(Image.open(B_path).convert('RGB'))
                imgs = self.crop({'I': B_img, 'T': A1_img})
                A1_img, B_img = Image.fromarray(imgs['T']), Image.fromarray(imgs['I'])
                is_natural_int = 1
            else:
                A1_path = self.A1_paths[index % self.A1_size]  # make sure index is within then range
                index_A2 = random.randint(0, self.A2_size - 1)
                A2_path = self.A2_paths[index_A2]
                B_path = ''

                A1_img = Image.open(A1_path).convert('RGB')
                A2_img = Image.open(A2_path).convert('RGB')
                B_img = Image.fromarray(np.zeros_like(A1_img))
                is_natural_int = 0
        else:  # test
            B_path = self.B_paths[index]
            B_img = Image.open(B_path).convert('RGB')
            if index < len(self.A1_paths):
                A1_path = self.A1_paths[index]
                A1_img = Image.open(A1_path).convert('RGB')
            else:
                A1_img = Image.fromarray(np.zeros_like(B_img))
            A2_img = None
            is_natural_int = 1

        w, h = A1_img.size
        neww = w // 4 * 4
        newh = h // 4 * 4
        resize = transforms.Resize([newh, neww])
        A1_img = resize(A1_img)
        A2_img = resize(A2_img) if A2_img else None
        B_img = resize(B_img)

        A1 = self.transform_A(A1_img)
        A2 = self.transform_A(A2_img) if A2_img else None
        B = self.transform_B(B_img)
        T2 = self.trans2(A1_img)
        T4 = self.trans4(A1_img)
        if A2 is not None:
            return {'T': A1, 'T2': T2, 'T4': T4, 'R': A2, 'I': B, 'B_paths': B_path, 'isNatural': is_natural_int}
        else:
            return {'T': A1, 'T2': T2, 'T4': T4, 'I': B, 'B_paths': B_path, 'isNatural': is_natural_int}

    def __len__(self):
        """Return the total number of images."""
        if self.opt.dataset_size == 0 or self.opt.phase == 'test':
            length = max(self.A1_size, self.A2_size, self.B_size)
        else:
            length = self.opt.dataset_size
        return length

