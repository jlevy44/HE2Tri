from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image

import os
import numpy as np
import cv2


class WSIDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt, bw_threshold=220):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.bw_threshold = bw_threshold
        self.img_path = os.path.join(opt.dataroot, opt.phase, opt.wsi_name)
        self.img = np.load(self.img_path)
        print("WSI image shape", self.img.shape)
        if opt.bgr2rgb:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        self.img_bw = (self.img_gray >= self.bw_threshold).astype(np.uint8)
        self.mask = np.zeros_like(self.img_bw, dtype=np.uint8)
        self.img_new = np.zeros_like(self.img, dtype=np.uint32)
        if opt.dps == 0:
            opt.dps = opt.load_size

        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        # self.transform = get_transform(opt, grayscale=(input_nc == 1))
        self.transform = get_transform(opt, grayscale=(input_nc == 1), to_pil=True)

        self.init_patch_info(opt)

    def init_patch_info(self, opt):
        self.patch_info = []
        count = 0
        for i in range(0, self.img.shape[0] - (opt.load_size - 1), opt.dps):
            for j in range(0, self.img.shape[1] - (opt.load_size - 1), opt.dps):
                if self.check_patch(i, j, i + opt.load_size, j + opt.load_size):
                    self.patch_info += [(i, j)]
                    count += 1
        numX = (self.img.shape[0] - (opt.load_size - 1)) // opt.dps + 1
        numY = (self.img.shape[1] - (opt.load_size - 1)) // opt.dps + 1
        print("Num patches:", count, "(" + str(numX) + " * " + str(numY) + " = " + str(numX * numY) + ")")

    def check_patch(self, x1, y1, x2, y2, max_brightness=0.92):
        return self.img_bw[x1:x2, y1:y2].mean() <= max_brightness

    def reset(self):
        self.mask = np.zeros_like(self.img_bw, dtype=np.uint8)
        self.img_new = np.zeros_like(self.img, dtype=np.uint32)


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        x1, y1 = self.patch_info[index]
        x2, y2 = x1 + self.opt.load_size, y1 + self.opt.load_size
        A = self.img[x1:x2, y1:y2]
        A = self.transform(A)
        return A

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.patch_info)

    def push_image(self, index, patch_img):
        x1, y1 = self.patch_info[index]
        x2, y2 = x1 + self.opt.load_size, y1 + self.opt.load_size
        self.img_new[x1:x2, y1:y2] += patch_img
        self.mask[x1:x2, y1:y2] += 1

    def apply_mask(self):
        temp_bw = np.stack((self.img_bw, self.img_bw, self.img_bw), axis=-1)
        temp_mask = self.mask + (self.mask == 0)
        temp_mask = np.stack((temp_mask, temp_mask, temp_mask), axis=-1)
        return ((temp_bw == 0) * (self.img_new // temp_mask) + temp_bw * self.img).astype(np.uint8)
