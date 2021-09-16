"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
	-- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
	-- <__len__>:                       return the size of dataset.
	-- <__getitem__>:                   get a data point from data loader.
	-- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
import numpy as np
import cv2, os
from scipy.ndimage import label as scilabel, distance_transform_edt
import scipy.ndimage as ndimage
from skimage import morphology as morph
from scipy.ndimage.morphology import binary_fill_holes as fill_holes
from .base_dataset import get_transform
from PIL import Image
from skimage.io import imread

def find_dataset_using_name(dataset_name):
	"""Import the module "data/[dataset_name]_dataset.py".

	In the file, the class called DatasetNameDataset() will
	be instantiated. It has to be a subclass of BaseDataset,
	and it is case-insensitive.
	"""
	dataset_filename = "data." + dataset_name + "_dataset"
	datasetlib = importlib.import_module(dataset_filename)

	dataset = None
	target_dataset_name = dataset_name.replace('_', '') + 'dataset'
	for name, cls in datasetlib.__dict__.items():
		if name.lower() == target_dataset_name.lower() \
		   and issubclass(cls, BaseDataset):
			dataset = cls

	if dataset is None:
		raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

	return dataset


def get_option_setter(dataset_name):
	"""Return the static method <modify_commandline_options> of the dataset class."""
	dataset_class = find_dataset_using_name(dataset_name)
	return dataset_class.modify_commandline_options


def create_dataset(opt):
	"""Create a dataset given the option.

	This function wraps the class CustomDatasetDataLoader.
		This is the main interface between this package and 'train.py'/'test.py'

	Example:
		>>> from data import create_dataset
		>>> dataset = create_dataset(opt)
	"""
	data_loader = CustomDatasetDataLoader(opt)
	dataset = data_loader.load_data()
	return dataset

def label_objects(I, min_object_size, threshold=220, connected_components=False, connectivity=8, kernel=8, apply_watershed=False):

	#try:
	BW = (cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)<threshold).astype(bool)
	#     if apply_watershed:
	#         BW = morph.binary_opening(BW, np.ones((connectivity,connectivity)).astype(int))
	labels = scilabel(BW)[0]
	if connected_components:
		BW = fill_holes(morph.remove_small_objects(labels, min_size=min_object_size, connectivity = connectivity, in_place=True))
		labels = scilabel(BW)[0]
	return(BW!=0),labels

def use_patch(mask,min_color=0.1):
	return (mask).astype(float).mean()>=min_color

def get_patch_info(he,basename,mask,patch_size=512,dps=512,threshold=220):
	import progressbar
	mask2=np.zeros_like(mask,dtype=np.single)
	patch_info=[]
	for x,y in progressbar.progressbar([(x1,y1) for x1 in np.arange(0,he.shape[0]-patch_size,dps) for y1 in np.arange(0,he.shape[1]-patch_size,dps)]):
		patch=he[x:x+patch_size,y:y+patch_size]
		if use_patch(mask[x:x+patch_size,y:y+patch_size],0.08):
			mask2[x:x+patch_size,y:y+patch_size]=mask2[x:x+patch_size,y:y+patch_size]+1
			patch_info.append([basename,x,y,patch_size])
	# mask2[mask2!=0]=1./mask2[mask2!=0]
	return patch_info, mask2


class NPYDataset_old(torch.utils.data.Dataset):
	def __init__(self,npy,bgr2rgb,opt,compression_factor=2.):
		#assert compression_factor==1. # figure out!!!
		self.compression_factor=compression_factor
		self.img=np.load(npy) if npy.endswith('.npy') else imread(npy)
		if self.compression_factor>1.:
			self.img=cv2.resize(self.img,None,fx=1./compression_factor,fy=1./compression_factor,interpolation=cv2.INTER_CUBIC)
		if bgr2rgb:
			self.img=cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
		self.mask=label_objects(self.img, 80000)[0]
		self.basename = os.path.basename(npy).replace('.npy','')
		if opt.dps==0:
			opt.dps=opt.load_size
		self.patch_info, self.mask = get_patch_info(self.img,self.basename,self.mask,patch_size=opt.load_size,dps=opt.dps,threshold=220)
		self.opt=opt
		input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
		# transform_params = get_params(self.opt, A.size)
		self.transform = get_transform(opt, grayscale=False, to_pil=True)#(np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
		print(self.transform)
		self.length=len(self.patch_info)
		print(self.length)


	def __getitem__(self,idx):
		basename,x,y,patch_size=self.patch_info[idx]
		img=self.img[x:x+patch_size,y:y+patch_size]#Image.fromarray()
		return self.transform(img)

	def __len__(self):
		return self.length

	def generate_image(self):
		self.img_new=np.zeros_like(self.img,dtype=np.uint8)
		# self.mask2=self.mask2.astype(float)

	def push_image(self, i,img):

		#for i,img in enumerate(imgs):
		basename,x,y,patch_size=self.patch_info[i]
		for i in range(3):
			self.img_new[x:x+patch_size,y:y+patch_size,i]=self.img_new[x:x+patch_size,y:y+patch_size,i]+(img[...,i]/self.mask[x:x+patch_size,y:y+patch_size]).astype(np.uint8)

		# return img_new

	def return_img(self, replace_original=True):
		# for i in range(3):
		# 	img_new[...,i]=img_new[...,i]/self.mask2
		self.img_new[self.mask==0,:]=255 if not replace_original else self.img[self.mask==0,:]
		return self.img_new#.astype(np.uint8)


class CustomDatasetDataLoader():
	"""Wrapper class of Dataset class that performs multi-threaded data loading"""

	def __init__(self, opt):
		"""Initialize this class

		Step 1: create a dataset instance given the name [dataset_mode]
		Step 2: create a multi-threaded data loader.
		"""
		self.opt = opt
		dataset_class = find_dataset_using_name(opt.dataset_mode)
		self.dataset = dataset_class(opt)
		print("dataset [%s] was created" % type(self.dataset).__name__)
		self.dataloader = torch.utils.data.DataLoader(
			self.dataset,
			batch_size=opt.batch_size,
			shuffle=not opt.serial_batches,
			num_workers=int(opt.num_threads))

	def load_data(self):
		return self

	def __len__(self):
		"""Return the number of data in the dataset"""
		return min(len(self.dataset), self.opt.max_dataset_size)

	def __iter__(self):
		"""Return a batch of data"""
		for i, data in enumerate(self.dataloader):
			if i * self.opt.batch_size >= self.opt.max_dataset_size:
				break
			yield data
