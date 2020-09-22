import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class ColorOnlyDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(input_nc=1, output_nc=3, display_ncols=3)
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        assert(self.opt.input_nc == 1)
        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        if self.opt.direction == 'BtoA':
            A = AB.crop((0, 0, w2, h)) # left color
            B = A.convert('L') # left gray
            A_nc = self.output_nc
            B_nc = self.input_nc
        else: # AtoB
            B = AB.crop((w2, 0, w, h)) # right color
            A = B.convert('L') # right gray
            A_nc = self.input_nc
            B_nc = self.output_nc

        # apply the same transform to A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(A_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(B_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        # C is other image
        if self.opt.direction == 'BtoA':
            C = AB.crop((w2, 0, w, h)) # right color
        else:
            C = AB.crop((0, 0, w2, h)) # left color
        gray_C = C.convert('L')
        C_nc = self.output_nc
        gray_C_nc = self.input_nc

        C_transform = get_transform(self.opt, transform_params, grayscale=(C_nc == 1))
        gray_C_transform = get_transform(self.opt, transform_params, grayscale=(gray_C_nc == 1))

        C = C_transform(C)
        gray_C = gray_C_transform(gray_C)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path, 'C': C, 'gray_C': gray_C}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
