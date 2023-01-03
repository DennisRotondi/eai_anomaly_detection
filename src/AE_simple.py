import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.utils
import wandb
import numpy as np
import pytorch_lightning as pl
from .data_module import MVTec_DataModule
import random
from torchmetrics import Recall, Precision, F1Score

def conv_block(in_features, out_features, kernel_size, stride, padding, bias, slope, normalize = True, affine = True):
	layer = [nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
	if normalize:
		layer += [nn.InstanceNorm2d(out_features, affine=affine)]
	if slope != 0:
		layer += [nn.LeakyReLU(slope, inplace = True)]
	return layer

def deconv_block(in_features, out_features, kernel_size, stride, padding, bias, slope, normalize = True, affine = True):
	layer = [nn.ConvTranspose2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
	if normalize:
		layer += [nn.BatchNorm2d(out_features, affine=affine)]
	if slope != 0:
		layer += [nn.LeakyReLU(slope, inplace = True)]
	return layer

class Encoder(nn.Module):
	def __init__(self, latent_dim):
		super().__init__()
		""" the input image is 3x256x256, this size allow to preserve the high res + still load all the dataset"""
		self.convolutions = nn.Sequential(
			*conv_block(3, 16, kernel_size=4, stride=2, padding=1, bias=False, slope = 0.19, normalize=True),
			*[m for mod in [conv_block(2**i, 2**(i+1), kernel_size=4, stride=2, padding=1, bias=False, slope = 0.19, normalize=True) \
				for i in range(4,9)] for m in mod],
			*conv_block(512, 1024, kernel_size=4, stride=1, padding=0, bias=False, slope = 0, normalize=False)
		)
		self.fc = nn.Linear(1024, latent_dim)

	def forward(self, img):
		return self.fc(self.convolutions(img).squeeze(-1).squeeze(-1))

class Decoder(nn.Module):
	def __init__(self, latent_dim):
		super().__init__()
		self.deconvolution = nn.Sequential(
			*deconv_block(latent_dim, 1024, kernel_size=4, stride=1, padding=0, bias=False, slope = 0.19, normalize=True),
			*[m for mod in [deconv_block(2**(i+1), 2**i, kernel_size=4, stride=2, padding=1, bias=False, slope = 0.19, normalize=True) \
				for i in range(9,4,-1)] for m in mod],
			*deconv_block(32, 3, kernel_size=4, stride=2, padding=1, bias=False, slope = 0, normalize=False))

	def forward(self, input):
		return torch.tanh(self.deconvolution(input.unsqueeze(-1).unsqueeze(-1)))

class AE(pl.LightningModule):
	""" Simple Autoencoder """
	def __init__(self, hparams):
		super(AE, self).__init__()
		self.save_hyperparameters(hparams)
		self.encoder = Encoder(self.hparams.latent_size)
		self.decoder = Decoder(self.hparams.latent_size)
		
		self.threshold = self.hparams.threshold # find a way to compute the "ideal" one!
		self.avg_anomaly = 0 # average anomaly score
		self.std_anomaly = 0 # standard deviation anomaly score

		self.val_precision = Precision(task = 'binary', num_classes = 2, average = 'macro')
		self.val_recall = Recall(task = 'binary', num_classes = 2, average = 'macro')
		self.val_f1score = F1Score(task = 'binary', num_classes = 2, average = 'macro')

	def forward(self, img):
		return self.decoder(self.encoder(img))
	
	def anomaly_score(self, img, recon): # (batch, 3, 256, 256)
		"""
		find a way to have this measure to output a value between 0 and 1!
		maybe... since the values of images are between [-1, 1] the maximum distance
		between two pixels is 2. So the maximum anomaly score that two images can obtain 
		in our setting is 2x3x256x256 = 393216.
		We normalize the result by dividing it by this value!
		"""
		recon = recon.view(recon.shape[0],-1)
		img = img.view(img.shape[0],-1)

		return (torch.abs(recon-img).sum(-1)) #/ 393216
	
	def anomaly_prediction(self, img, recon):
		anomaly_score = self.anomaly_score(img, recon)
		ris = (anomaly_score > self.threshold).long()
		return ris

	def configure_optimizers(self):
		# note wd = 0 in this simplest version (baseline)
		optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, eps=self.hparams.adam_eps, weight_decay=0)
		reduce_lr_on_plateau = ReduceLROnPlateau(optimizer, mode='min',verbose=True, min_lr=self.hparams.min_lr)
		return {
			"optimizer": optimizer,
			"lr_scheduler": {
				"scheduler": reduce_lr_on_plateau,
				"monitor": 'loss',
				"frequency": 1
			},
		}

	def loss_function(self,recon_x, x):
		""" loss function is mse, 100* (possible hp) is to enlarge the recon diff """
		return {"loss": self.hparams.loss_weight*F.mse_loss(recon_x, x, reduction='sum')}

	def training_step(self, batch, batch_idx):
		imgs = batch['img']
		recon = self(imgs)
		loss = self.loss_function(recon, imgs)
		# LOSS
		self.log_dict(loss)
		# ANOMALY SCORE --> mean and standard deviation
		# in addition to the loss we're going to compute the "anomaly score", that's not necessarily the same measure of the loss.
		anomaly_scores = self.anomaly_score(imgs, recon)
		a_mean = anomaly_scores.mean().detach().cpu().numpy()
		a_std = anomaly_scores.std().detach().cpu().numpy()
		return {'loss': loss['loss'], 'anom': a_mean, 'a_std': a_std}

	def training_epoch_end(self, outputs):
		a = np.stack([x['anom'] for x in outputs]) 
		a_std = np.stack([x['a_std'] for x in outputs]) 
		self.avg_anomaly = a.mean()
		self.std_anomaly = a_std.mean()
		self.log_dict({"anom_avg": self.avg_anomaly, "anom_std": self.std_anomaly})
		# THRESHOLD UPDATE
		# https://www.mdpi.com/1424-8220/22/8/2886
		# more conservative approach to improve RECALL if w_std == -1
		self.threshold = (1-self.hparams.t_weight)*self.threshold + \
							self.hparams.t_weight*(self.avg_anomaly + self.hparams.w_std*self.std_anomaly)
		self.log("anomaly_threshold", self.threshold, on_step=False, on_epoch=True, prog_bar=True)

	# images logging during training phase but used for validation images
	def get_images_for_log(self, real, reconstructed):
		example_images = []
		real = MVTec_DataModule.denormalize(real)
		reconstructed = MVTec_DataModule.denormalize(reconstructed)
		for i in range(real.shape[0]):
			couple = torchvision.utils.make_grid(
				[real[i], reconstructed[i]],
				nrow=2,
				# normalize=True,
				scale_each=False,
				pad_value=1,
				padding=4,
			)  
			example_images.append(
				wandb.Image(couple.permute(1, 2, 0).detach().cpu().numpy(), mode="RGB")
			)
		return example_images

	def validation_step(self, batch, batch_idx):
		imgs = batch['img']
		recon_imgs = self(imgs)
		# LOSS
		self.log("val_loss", self.loss_function(recon_imgs, imgs)["loss"], on_step=False, on_epoch=True, batch_size=imgs.shape[0])
		# RECALL, PRECISION, F1 SCORE
		pred = self.anomaly_prediction(imgs, recon_imgs)
		self.log("precision", self.val_precision(pred, batch['label']), on_step=False, on_epoch=True, prog_bar=True, batch_size=imgs.shape[0])
		self.log("recall", self.val_recall(pred, batch['label']), on_step=False, on_epoch=True, prog_bar=True, batch_size=imgs.shape[0])
		self.log("f1_score", self.val_f1score(pred, batch['label']), on_step=False, on_epoch=True, prog_bar=True, batch_size=imgs.shape[0])
		# IMAGES
		images = self.get_images_for_log(imgs[0:self.hparams.log_images], recon_imgs[0:self.hparams.log_images])
		return {"images": images}

	def validation_epoch_end(self, outputs):
		if self.global_step%self.hparams.log_image_each_epoch==0:
			# we randomly select one batch index
			bidx = random.randrange(100) % len(outputs)
			images = outputs[bidx]["images"]
			self.logger.experiment.log({f"images": images})