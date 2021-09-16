"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
	Test a CycleGAN model (both sides):
		python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

	Test a CycleGAN model (one side only):
		python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

	The option '--model test' is used for generating CycleGAN results only for one side.
	This option will automatically set '--dataset_mode single', which only loads the images from one set.
	On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
	which is sometimes unnecessary. The results will be saved at ./results/.
	Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

	Test a pix2pix model:
		python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_wsi_options import TestWSIOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import numpy as np
import cv2
import subprocess
import time
import tqdm

if __name__ == '__main__':
	PROGRAM_START_TIME = time.time()
	opt = TestWSIOptions().parse()  # get test options
	# hard-code some parameters for test
	opt.num_threads = 0   # test code only supports num_threads = 1
	opt.batch_size = 1    # test code only supports batch_size = 1
	opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
	opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
	opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
	dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

	total_iter = opt.load_iter + 1
	start_iter = opt.iter_start
	if opt.iter_start < 0:
		opt.iter_start = opt.load_iter
		opt.iter_incr = 1
	model = create_model(opt)
	if opt.dataset_mode=="wsi":
		for iter in range(start_iter, total_iter, opt.iter_incr):
			opt.load_iter = iter
			print("iter", opt.load_iter)
			ITER_START_TIME = time.time()
			dataset.dataset.reset()
			# create save location for results
			subfolder_name = '{}_{}'.format(opt.phase, opt.epoch)
			if True: # opt.load_iter > 0:  # load_iter is 0 by default
				subfolder_name = '{:s}_iter{:d}'.format(subfolder_name, opt.load_iter)
			web_dir = os.path.join(opt.results_dir, opt.name, subfolder_name)
			new_wsi_filename = opt.wsi_name.replace('.npy', '_converted.npy')
			save_path = os.path.join(web_dir, "images", new_wsi_filename)
			print('save_path', save_path)
			print('creating web directory', web_dir)
			webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
	      # create a model given opt.model and other options
			model.setup(opt)               # regular setup: load and print networks; create schedulers
			# test with eval mode. This only affects layers like batchnorm and dropout.
			# For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
			# For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
			if opt.eval:
				model.eval()

			output=[]
			for i, data in enumerate(dataset):
				model.set_input(data)  # unpack data from data loader
				model.test()           # run inference
				img=((model.fake.detach().cpu().numpy()[0].transpose((1,2,0)) + 1.) / 2. * 255.).astype(np.uint8)
				dataset.dataset.push_image(i, img)
				# if i % 50 == 0:
				# 	print('processing {} - th patch'.format(i))
			img_new = dataset.dataset.apply_mask()
			np.save(save_path, img_new)
			# subprocess.call("python npy2dzi.py --wsi_name {} --web_dir {} --shrink_factor {}".format(new_wsi_filename, web_dir, opt.shrink_factor), shell=True)
			print("Iter execution time (s)", time.time() - ITER_START_TIME)
	elif opt.dataset_mode=="npy":
		# opt.load_iter = iter
		# print("iter", opt.load_iter)
		ITER_START_TIME = time.time()
		# dataset.dataset.reset()
		# create save location for results
		# subfolder_name = '{}_{}'.format(opt.phase, opt.epoch)
		# if True: # opt.load_iter > 0:  # load_iter is 0 by default
		# 	subfolder_name = '{:s}_iter{:d}'.format(subfolder_name, opt.load_iter)
		# web_dir = os.path.join(opt.results_dir, opt.name, subfolder_name)
		save_path = os.path.join(opt.results_dir_wsi,os.path.basename(opt.wsi_name.replace('.npy', '_converted.npy')))
		# save_path = os.path.join(web_dir, "images", new_wsi_filename)
		# print('save_path', save_path)
		# print('creating web directory', web_dir)
		# webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
		# model = create_model(opt)      # create a model given opt.model and other options
		model.setup(opt)               # regular setup: load and print networks; create schedulers
		# test with eval mode. This only affects layers like batchnorm and dropout.
		# For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
		# For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
		if opt.eval:
			model.eval()

		output=[]
		print(dir(dataset))
		print(dataset.dataset)
		for i, data in tqdm(enumerate(dataset.dataset),total=len(dataset.dataset)):
			model.set_input(data)  # unpack data from data loader
			model.test()           # run inference
			img=((model.fake.detach().cpu().numpy()[0].transpose((1,2,0)) + 1.) / 2. * 255.).astype(np.uint8)
			dataset.dataset.push_image(i, img)
			# if i % 50 == 0:
			# 	print('processing {} - th patch'.format(i))
		img_new = dataset.dataset.img_new
		np.save(save_path, img_new)
		# subprocess.call("python npy2dzi.py --wsi_name {} --web_dir {} --shrink_factor {}".format(new_wsi_filename, web_dir, opt.shrink_factor), shell=True)
		print("Iter execution time (s)", time.time() - ITER_START_TIME)

	print("Total execution time (s)", time.time() - PROGRAM_START_TIME)
