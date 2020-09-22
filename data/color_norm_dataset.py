import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class ColorNormDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(input_nc=1, output_nc=3, display_ncols=6)
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

        A_rgb = AB.crop((0, 0, w2, h))
        A_gray = A_rgb.convert('L')
        B_rgb = AB.crop((w2, 0, w, h))
        B_gray = B_rgb.convert('L')
        rgb_nc = self.output_nc
        gray_nc = self.input_nc

        # apply transforms to A and B
        transform_params = get_params(self.opt, A_rgb.size)
        rgb_transform = get_transform(self.opt, transform_params, grayscale=(rgb_nc == 1))
        gray_transform = get_transform(self.opt, transform_params, grayscale=(gray_nc == 1))

        A_rgb = rgb_transform(A_rgb)
        A_gray = gray_transform(A_gray)
        B_rgb = rgb_transform(B_rgb)
        B_gray = gray_transform(B_gray)

        return {'A_rgb': A_rgb, 'A_gray': A_gray, 'B_rgb': B_rgb, 'B_gray': B_gray, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
