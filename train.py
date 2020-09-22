"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
	Train a CycleGAN model:
		python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
	Train a pix2pix model:
		python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time, numpy as np
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import subprocess, os

if __name__ == '__main__':
	opt = TrainOptions().parse()   # get training options
	dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
	dataset_size = len(dataset)    # get the number of images in the dataset.
	print('The number of training images = %d' % dataset_size)

	model = create_model(opt)      # create a model given opt.model and other options
	model.setup(opt)               # regular setup: load and print networks; create schedulers
	visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
	total_iters = 0                # the total number of training iterations

	for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
		epoch_start_time = time.time()  # timer for entire epoch
		iter_data_time = time.time()    # timer for data loading per iteration
		epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
		visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

		for i, data in enumerate(dataset):  # inner loop within one epoch
			iter_start_time = time.time()  # timer for computation per iteration
			if total_iters % opt.print_freq == 0:
				t_data = iter_start_time - iter_data_time

			total_iters += opt.batch_size
			epoch_iter += opt.batch_size
			model.set_input(data)         # unpack data from dataset and apply preprocessing
			model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

			if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
				save_result = total_iters % opt.update_html_freq == 0
				model.compute_visuals()
				visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

			if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
				losses = model.get_current_losses()
				t_comp = (time.time() - iter_start_time) / opt.batch_size
				visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
				if opt.display_id > 0:
					visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

			if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
				print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
				save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
				model.save_networks(save_suffix)

			iter_data_time = time.time()

			# uncomment and manipulate to export dzi images real time
			'''if ((total_iters-opt.batch_size==0) or (total_iters % opt.test_wsi_iters == 0)) and opt.rsync_destination: # 0, 133
				basenames=[BASENAMES]
				save_suffix = 'iter_%d' % total_iters
				custom_load_size=512
				custom_dps=256
				is_p2p=True
				compression_factor=1.#2.
				model.save_networks(save_suffix)
				dataset_name=opt.name
				webserver_dir = "X"
				output_dir = "Y'
				input_dir = "Z'
				test_files=[]
				for basename in basenames:
					output_basename='{}.iter.{}.npy'.format(basename,total_iters)
					test_files.append(os.path.join(output_dir,output_basename))
					subprocess.call("python test_wsi.py --output_dir {0} --output_basename {1} --dataroot datasets/{2} --name {2} --no_dropout --model_loc G:{3} --num_test 1000000  --npy {4}.npy --preprocess scale_width --load_size {5} --dps {6} --model {6} {7} ".format(output_dir,output_basename,dataset_name,'./checkpoints/{}/{}_net_G{}.pth'.format(dataset_name,save_suffix,"" if is_p2p else "_A"),os.path.join(input_dir,str(basename)),custom_load_size,custom_dps,"test" if not is_p2p else "test","" if not is_p2p else "--netG unet_256 --norm batch"),shell=True)
				for test_file in test_files:
					subprocess.call("python npy2dzi_orig.py --npy {} --out_dir {} --compression_factor {}".format(test_file,webserver_dir,compression_factor),shell=True)
			'''

		if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
			print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
			model.save_networks('latest')
			model.save_networks(epoch)

		print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
		model.update_learning_rate()                     # update learning rates at the end of every epoch.
