from typing import Any, Union, List, Optional
import os
from tqdm import tqdm
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

class MVTec_Dataset(Dataset):
	def __init__(self, dataset_dir: str, train_or_test: str, hparams: Any):
		self.data = list() # list of images with their class and label (0 normal, 1 anomalous)
		self.train_or_test = train_or_test
		self.dataset_dir = dataset_dir
		self.hparams = hparams
		self.transform = transforms.Compose([ # classica trasformazione per immagini
			transforms.Resize((hparams.img_size, hparams.img_size)),
			# Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] 
			# to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
			transforms.ToTensor(),
			# to have 0 mean and values in range [-1, 1]
			# The parameters mean, std are passed as 0.5, 0.5 in your case. 
			# This will normalize the image in the range [-1,1]. For example,
			# the minimum value 0 will be converted to (0-0.5)/0.5=-1, 
			# the maximum value of 1 will be converted to (1-0.5)/0.5=1.
			# https://discuss.pytorch.org/t/understanding-transform-normalize/21730
			transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
		])
		self.make_data()

	def make_data(self):
		if self.hparams.data_loading_strategy=="normal":
			self.make_data_1()
		else:
			self.make_data_2()
	
	def make_data_1(self):
		# this function read the fresh downloaded dataset and make it ready for the training
		class_dir_list = list()
		for f in [os.path.join(self.dataset_dir, e) for e in os.listdir(self.dataset_dir)]:
			if os.path.isdir(f):
				class_dir_list.append(f)
		for f in class_dir_list:
			class_obj = f.split("/")[-1]
			print("["+class_obj+"]")
			for dir in os.listdir(f):
				if dir==self.train_or_test:
					print("## "+dir+" ##")
					current_dir = os.path.join(f,dir)
					for t in os.listdir(current_dir):
						imgs = os.path.join(current_dir,t)
						label = 1 if t=="good" else 0
						for image_path in tqdm([os.path.join(imgs,e) for e in os.listdir(imgs)]):
							img = self.transform(Image.open(image_path).convert('RGB'))
							self.data.append({"img" : img, "class_obj": class_obj, "label" : label})
	
	def make_data_2(self):
		"""
		We tried this additional data extraction strategy in order to make data.setup() more efficient!
        We thought the slowness of the operation was induced by the many folder accesses and as a result
        we the dataset folder structure is been modified. NO IMPROVEMENTS were achieved. 
        The lack of efficiency comes from the image transformations!
		"""
		for dir in os.listdir(self.dataset_dir):
			if dir==self.train_or_test:
				current_dir = os.path.join(self.dataset_dir, dir)
				for t in os.listdir(current_dir):
					label = 1 if t=="good" else 0
					imgs = os.path.join(current_dir,t)
					print("["+imgs.split("/")[-2]+"/"+imgs.split("/")[-1]+"]")
					for image_path in tqdm([os.path.join(imgs,e) for e in os.listdir(imgs)]):
							img = self.transform(Image.open(image_path).convert('RGB'))
							class_obj = (image_path.split("/")[-1]).split("_")[0]
							self.data.append({"img" : img, "class_obj": class_obj, "label" : label})
	
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]

class MVTec_DataModule(pl.LightningDataModule):
	def __init__(self, hparams: dict):
		super().__init__()
		self.save_hyperparameters(hparams)

	def setup(self, stage: Optional[str] = None) -> None:
		# TRAIN
		self.data_train = MVTec_Dataset(self.hparams.dataset_dir, "train", self.hparams)
		# TEST
		self.data_test = MVTec_Dataset(self.hparams.dataset_dir, "test", self.hparams)

	def train_dataloader(self):
		return DataLoader(
			self.data_train,
			batch_size=self.hparams.batch_size,
			shuffle=True,
			num_workers=self.hparams.n_cpu,
			pin_memory=self.hparams.pin_memory,
			persistent_workers=True
		)

	def val_dataloader(self):
		return DataLoader(
			self.data_test,
			batch_size=self.hparams.batch_size,
			shuffle=False,
			num_workers=self.hparams.n_cpu,
			pin_memory=self.hparams.pin_memory,
			persistent_workers=True
		)
	#to invert the normalization of the compose transform.
	@staticmethod
	def denormalize(tensor):
		return tensor*0.5 + 0.5