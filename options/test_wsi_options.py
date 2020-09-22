from .base_options import BaseOptions


class TestWSIOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options

        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')

        # PATHFLOW
        parser.add_argument('--wsi_name', type=str, default='', help='filename of the WSI image.')
        parser.add_argument('--shrink_factor', type=int, default=1, help='the factor that the WSI image will be reduced by.')
        parser.add_argument('--bgr2rgb', action='store_true', help='change color.')
        parser.add_argument('--iter_start', type=int, default=-1, help='increments of iterations')
        parser.add_argument('--iter_incr', type=int, default=1, help='increments of iterations')

        # rewrite devalue values
        parser.set_defaults(model='test')
        parser.set_defaults(dataset_mode='wsi')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser

    def parse(self):
        opt = BaseOptions.parse(self)
        assert(opt.crop_size == opt.load_size)
        assert(opt.model=="test")
        return opt
