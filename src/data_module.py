from typing import Any, Union, List, Optional
import os
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
        # this function read the fresh downloaded dataset and make it ready for the training
		class_dir_list = list()
		for f in [os.path.join(self.dataset_dir, e) for e in os.listdir(self.dataset_dir)]:
			if os.path.isdir(f):
				class_dir_list.append(f)
		for f in class_dir_list:
			class_obj = f.split("/")[-1]
			for dir in os.listdir(f):
				if dir==self.train_or_test:
					current_dir = os.path.join(f,dir)
					for t in os.listdir(current_dir):
						imgs = os.path.join(current_dir,t)
						label = 1 if t=="good" else 0
						for image_path in [os.path.join(imgs,e) for e in os.listdir(imgs)]:
							img = self.transform(Image.open(image_path).convert('RGB'))
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
            # collate_fn=self.collate_train,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.n_cpu,
            # collate_fn=self.collate_test,
            pin_memory=True,
            persistent_workers=True
        )
    #to invert the normalization of the compose transform.
    @staticmethod
    def denormalize(tensor):
        return tensor*0.5 + 0.5    
    # def collate_train(self, batch):
    #     batch_out = dict()
    #     batch_out["id"] = [sample["id"] for sample in batch]
    #     batch_out["img"] = torch.stack([sample["img"] for sample in batch], dim=0)
    #     return batch_out
    
    # def collate_test(self, batch):
    #     batch_out = dict()
    #     batch_out["id"] = [sample["id"] for sample in batch]
    #     batch_out["img"] = torch.stack([sample["img"] for sample in batch], dim=0) 
    #     batch_out["label"] = [sample["label"] for sample in batch]
    #     return batch_out