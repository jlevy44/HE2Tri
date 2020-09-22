import torch
import torchvision
from .base_model import BaseModel
from . import networks
from . import losses
import kornia

import itertools
from util.image_pool import ImagePool



class ColorNormModel(BaseModel):
    """
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='instance', dataset_mode='color_norm')
        # default cg
        parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        # extra options
        parser.add_argument('--crop_border', type=int, default=0, help='ignores the border of the image when calculating loss')
        if is_train:
            # default cg
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            # default p2p
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--enable_D_A2B', action='store_true', help='enable the use of A2B in discriminator')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # CycleGAN loss/visual names
        cg_loss_names_A = ['D_A', 'G_A', 'cycle_A']
        cg_loss_names_B = ['D_B', 'G_B', 'cycle_B']
        visual_names_A = ['real_A_rgb', 'real_A_gray', 'fake_B_gray', 'rec_A_gray']
        visual_names_B = ['real_B_rgb', 'real_B_gray', 'fake_A_gray', 'rec_B_gray']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used
            cg_loss_names_A += ['idt_B']
            cg_loss_names_B += ['idt_A']
            visual_names_A += ['idt_B_gray']
            visual_names_B += ['idt_A_gray']
        else:
            self.opt.display_ncols = 5
        # Pix2Pix loss/visual names
        p2p_loss_names = ['D', 'G_GAN', 'G_L1']
        visual_names_A += ['fake_A2B_rgb']
        visual_names_B += ['fake_B2B_rgb']

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = cg_loss_names_A + cg_loss_names_B + p2p_loss_names
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = visual_names_A + visual_names_B

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'G', 'D']
        else:  # during test time, only load generators
            self.model_names = ['G_A', 'G_B', 'G']

        # define networks (both generator and discriminator)

        # CycleGAN generators
        self.netG_A = networks.define_G(opt.input_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.input_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # Pix2Pix generators
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # CycleGAN discriminator
            self.netD_A = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # Pix2Pix discriminator
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # CycleGAN loss functions
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # Pix2Pix loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G_cg = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_cg = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G_cg)
            self.optimizers.append(self.optimizer_D_cg)

            self.optimizer_D_p2p = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_p2p = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G_p2p)
            self.optimizers.append(self.optimizer_D_p2p)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A_rgb = input['A_rgb' if AtoB else 'B_rgb'].to(self.device)
        self.real_A_gray = input['A_gray' if AtoB else 'B_gray'].to(self.device)
        self.real_B_rgb = input['B_rgb' if AtoB else 'A_rgb'].to(self.device)
        self.real_B_gray = input['B_gray' if AtoB else 'A_gray'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def crop_border(self, img):
        return img[:,:,self.opt.crop_border:-self.opt.crop_border,self.opt.crop_border:-self.opt.crop_border]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # CycleGAN
        self.fake_B_gray = self.netG_A(self.real_A_gray)    # G_A(A)
        self.rec_A_gray = self.netG_B(self.fake_B_gray)     # G_B(G_A(A))
        self.fake_A_gray = self.netG_B(self.real_B_gray)    # G_B(B)
        self.rec_B_gray = self.netG_A(self.fake_A_gray)     # G_A(G_B(B))

        if self.isTrain and self.opt.lambda_identity > 0.0:
            self.idt_A_gray = self.netG_A(self.real_B_gray) # G_A(B)
            self.idt_B_gray = self.netG_B(self.real_A_gray) # G_B(A)

        # Pix2Pix
        self.fake_B2B_rgb = self.netG(self.real_B_gray)     # G(real_B) or G(B)
        with torch.no_grad():
            self.fake_A2B_rgb = self.netG(self.fake_B_gray) # G(fake_B) or G(G_A(A))

        # crops borders before returning/calculating losses
        if self.opt.crop_border > 0:
            self.real_A_rgb = self.crop_border(self.real_A_rgb)
            self.real_A_gray = self.crop_border(self.real_A_gray)
            self.fake_B_gray = self.crop_border(self.fake_B_gray)
            self.rec_A_gray = self.crop_border(self.rec_A_gray)
            self.fake_A2B_rgb = self.crop_border(self.fake_A2B_rgb)

            self.real_B_rgb = self.crop_border(self.real_B_rgb)
            self.real_B_gray = self.crop_border(self.real_B_gray)
            self.fake_A_gray = self.crop_border(self.fake_A_gray)
            self.rec_B_gray = self.crop_border(self.rec_B_gray)
            self.fake_B2B_rgb = self.crop_border(self.fake_B2B_rgb)

            if self.isTrain and self.opt.lambda_identity > 0.0:
                self.idt_A_gray = self.crop_border(self.idt_A_gray)
                self.idt_B_gray = self.crop_border(self.idt_B_gray)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B_gray = self.fake_B_pool.query(self.fake_B_gray)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B_gray, fake_B_gray)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A_gray = self.fake_A_pool.query(self.fake_A_gray)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A_gray, fake_A_gray)

    def backward_G_cg(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.loss_idt_A = self.criterionIdt(self.idt_A_gray, self.real_B_gray) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.loss_idt_B = self.criterionIdt(self.idt_B_gray, self.real_A_gray) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B_gray), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A_gray), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A_gray, self.real_A_gray) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B_gray, self.real_B_gray) * lambda_B
        # combined loss and calculate gradients
        self.loss_G_cg = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G_cg.backward()


    def backward_D(self):
        loss_count = 2

        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_gray = torch.cat((self.real_B_gray, self.fake_B2B_rgb), 1)
        pred_fake = self.netD(fake_gray.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_gray = torch.cat((self.real_B_gray, self.real_B_rgb), 1)
        pred_real = self.netD(real_gray)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G_p2p(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_gray = torch.cat((self.real_B_gray, self.fake_B2B_rgb), 1)
        pred_fake = self.netD(fake_gray)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_L1 = 0
        # Second, G(A) = B
        if self.opt.lambda_L1 != 0:
            self.loss_G_L1 = self.criterionL1(self.fake_B2B_rgb, self.real_B_rgb) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G_p2p = self.loss_G_GAN + self.loss_G_L1
        self.loss_G_p2p.backward()

    def optimize_parameters(self):
        self.set_requires_grad([self.netG_A, self.netG_B, self.netD_A, self.netD_B, self.netG_A, self.netG_B], True)
        self.forward()                   # compute fake images: G(A)

        # CycleGAN optimize
        self.set_requires_grad([self.netD, self.netG], False)                               # Turn off Pix2Pix gradients
        # G_A and G_B
        self.set_requires_grad([self.netG_A, self.netG_B], True)
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G_cg.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G_cg()             # calculate gradients for G_A and G_B
        self.optimizer_G_cg.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netG_A, self.netG_B], False)
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D_cg.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D_cg.step()  # update D_A and D_B's weights

        # Pix2Pix optimize
        self.set_requires_grad([self.netG_A, self.netG_B, self.netD_A, self.netD_B], False) # Turn off CycleGAN gradients
        # update D
        self.set_requires_grad(self.netG, False)
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D_p2p.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D_p2p.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netG, True)
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G_p2p.zero_grad()        # set G's gradients to zero
        self.backward_G_p2p()                   # calculate graidents for G
        self.optimizer_G_p2p.step()             # udpate G's weights
