from typing import Any, Self, Union, List, Optional
import os

import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

# function for generating unique IDs!
def uniqueid():
	seed = 0
	while True:
		yield seed
		seed += 1

class MVTec:
	def __init__(self, images_folder: str, train_or_test: str):
		
		self.data = {}
		self.train_or_test = train_or_test
		id_generator = uniqueid()

		class_folder_list = []
		for f in [images_folder+e for e in os.listdir(images_folder)]: # "/home/lavallone/Desktop/EAI_Anomaly_Detection/data/"
			if os.path.isdir(f):
				class_folder_list.append(images_folder+f)
		
		for f in class_folder_list:
			for folder in os.listdir(f):
				if folder=="train" and self.train_or_test=="train":
					for image_path in [f+"/"+folder+"/good/"+e for e in os.listdir(f+"/"+folder+"/good/")]:
						self.data[str(next(id_generator))] = {"image_path" : image_path}
				if folder=="test" and self.train_or_test=="test":
					for t in os.listdir(f+"/"+folder):
						if t=="good":
							for image_path in [f+"/"+folder+"/"+t+"/"+e for e in os.listdir(f+"/"+folder+"/"+t+"/")]:
								self.data[str(next(id_generator))] = {"image_path" : image_path, "label" : 0}
						else:
							for image_path in [f+"/"+folder+"/"+t+"/"+e for e in os.listdir(f+"/"+folder+"/"+t+"/")]:
								self.data[str(next(id_generator))] = {"image_path" : image_path, "label" : 1}

	#def get_img_from_id(self, img_id):
	#	id = str(img_id)
	#	return self.data[id]["image_path"]


# train_MVTec = MVTec("/home/lavallone/Desktop/EAI_Anomaly_Detection/data/", "train")
# test_MVTec = MVTec("/home/lavallone/Desktop/EAI_Anomaly_Detection/data/", "train")

class MVTec_Dataset(Dataset):
	def __init__(self, MVTec):
		self.data = self.make_data(MVTec)

	def make_data(self, MVTec):
		transform = transforms.Compose([ # classica trasformazione per immagini
			transforms.Resize((224, 224)),
			transforms.ToTensor()
		])
		data = list()
		for k in MVTec.data:
			item = dict()
			item["id"] = k
			img_pth = MVTec.data[k]["image_path"]
			item["img"] = transform(Image.open(img_pth).convert('RGB'))
			if MVTec.train_or_test == "test":
				item["label"] = MVTec.data[k]["label"]
			data.append(item)
		return data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]


class MVTec_DataModule(pl.LightningDataModule):
    def __init__(self, hparams: dict, train_MVTec: Any, test_MVTec: Any) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.train_MVTec = train_MVTec
        self.test_MVTec = test_MVTec

    def setup(self, stage: Optional[str] = None) -> None:
        # TRAIN
        self.data_train = MVTec_Dataset(self.train_MVTec)
        # TEST
        self.data_test = MVTec_Dataset(self.test_MVTec)

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.n_cpu,
            collate_fn=self.collate_train,
            #pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.n_cpu,
            collate_fn=self.collate_test,
            #pin_memory=True,
            persistent_workers=True
        )
        
    def collate_train(self, batch):
        batch_out = dict()
        batch_out["id"] = [sample["id"] for sample in batch]
        batch_out["img"] = torch.stack([sample["img"] for sample in batch], dim=0)
        return batch_out
    
    def collate_test(self, batch):
        batch_out = dict()
        batch_out["id"] = [sample["id"] for sample in batch]
        batch_out["img"] = torch.stack([sample["img"] for sample in batch], dim=0) 
        batch_out["label"] = [sample["label"] for sample in batch]
        return batch_out
    
# data = VQA_DataModule(hparams, train_MVTec, test_MVTec)