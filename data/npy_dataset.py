from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image

import os
import numpy as np
import cv2


class NPYDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.img_path = os.path.abspath(opt.wsi_name)
        self.img = np.load(self.img_path)
        print("WSI image shape", self.img.shape)
        if False or opt.bgr2rgb:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        # self.transform = get_transform(opt, grayscale=(input_nc == 1))
        self.transform = get_transform(opt, grayscale=(input_nc == 1), to_pil=True)
        self.reset()

    def reset(self):
        self.img_new = np.zeros_like(self.img, dtype=np.uint32)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A = self.img[index]
        A = self.transform(A)
        return A

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.img.shape[0]

    def push_image(self, index, patch_img):
        self.img_new[index] = patch_img
