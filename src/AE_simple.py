from collections import Counter
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
from torchmetrics.functional import structural_similarity_index_measure as SSIM
from torchmetrics.functional import multiscale_structural_similarity_index_measure as MSSIM
from torchmetrics.classification import BinaryAUROC


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
	def __init__(self, latent_dim, channels, hparams):
		super().__init__()
		""" the input image is 3x256x256, this size allow to preserve the high res + still load all the dataset"""
		self.convolutions = nn.Sequential(
			*conv_block(channels, 16, kernel_size=4, stride=2, padding=1, bias=False, slope = hparams.slope, normalize=True),
			*[m for mod in [conv_block(2**i, 2**(i+1), kernel_size=4, stride=2, padding=1, bias=False, slope = hparams.slope, normalize=True) \
				for i in range(4,9)] for m in mod],
			*conv_block(512, 1024, kernel_size=4, stride=1, padding=0, bias=False, slope = 0, normalize=False)
		)
		self.fc = nn.Linear(1024, latent_dim)

	def forward(self, img):
		return self.fc(self.convolutions(img).squeeze(-1).squeeze(-1))

class Decoder(nn.Module):
	def __init__(self, latent_dim, channels, hparams):
		super().__init__()
		self.deconvolution = nn.Sequential(
			*deconv_block(latent_dim, 1024, kernel_size=4, stride=1, padding=0, bias=False, slope = hparams.slope, normalize=True),
			*[m for mod in [deconv_block(2**(i+1), 2**i, kernel_size=4, stride=2, padding=1, bias=False, slope = hparams.slope, normalize=True) \
				for i in range(9,4,-1)] for m in mod],
			*deconv_block(32, channels, kernel_size=4, stride=2, padding=1, bias=False, slope = 0, normalize=False))

	def forward(self, input):
		return torch.tanh(self.deconvolution(input.unsqueeze(-1).unsqueeze(-1)))

class AE(pl.LightningModule):
	""" Simple Autoencoder """
	def __init__(self, hparams):
		super(AE, self).__init__()
		self.save_hyperparameters(hparams)
		self.encoder = Encoder(self.hparams.latent_size, self.hparams.img_channels, self.hparams)
		self.decoder = Decoder(self.hparams.latent_size, self.hparams.img_channels, self.hparams)
		# https://pytorch.org/docs/master/generated/torch.nn.Module.html?highlight=apply#torch.nn.Module.apply
		if self.hparams.normalization:
			self.encoder.apply(self.weights_init_normal)
			self.decoder.apply(self.weights_init_normal)
		self.threshold = self.hparams.threshold # there are different ways to compute the "ideal" one!

		self.val_precision = Precision(task = 'binary', num_classes = 2, average = 'macro')
		self.val_recall = Recall(task = 'binary', num_classes = 2, average = 'macro')
		self.val_f1score = F1Score(task = 'binary', num_classes = 2, average = 'macro')
		self.val_auroc = BinaryAUROC()

	# to apply the weights initialization of cycle-gan paper
	# it ensures better performances 
	def weights_init_normal(self, m: nn.Module):
		classname = m.__class__.__name__
		if classname.find("Conv") != -1 or classname.find("Linear") != -1:
			torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
			if hasattr(m, "bias") and m.bias is not None:
				torch.nn.init.constant_(m.bias.data, 0.0)
		elif classname.find("Norm2d") != -1:
			torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
			torch.nn.init.constant_(m.bias.data, 0.0)

	def forward(self, img):
		return self.decoder(self.encoder(img))
	
	def anomaly_score(self, img, recon): # (batch, 3, 256, 256)
		"""
		The maximum anomaly score with MSE that two images 
		can obtain in our setting is 2x3x256x256 = 393216.
		We could even normalize the result by dividing it by this value!
		"""
		if self.hparams.anomaly_stategy == "mse":
			recon = recon.view(recon.shape[0],-1)
			img = img.view(img.shape[0],-1)
			return (torch.abs(recon-img).sum(-1)) #/ 393216
		elif self.hparams.anomaly_stategy == "ssim":
			return SSIM(img, recon, reduction=None)
		else:
			return MSSIM(img, recon, reduction=None)
	
	def anomaly_prediction(self, img, recon=None):
		if recon is None:
			recon = self(img)
		anomaly_score = self.anomaly_score(img, recon)
		ris = (anomaly_score > self.threshold).long()
		return ris

	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, eps=self.hparams.adam_eps, weight_decay=self.hparams.wd)
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
		# note we are using only mse, no contractive effort in the simplest version
		loss = self.hparams.loss_weight*(recon_x-x)**2
		if self.hparams.reduction == 'mean':
			loss = loss.mean()
		else:
			loss = loss.sum()
		return {"loss": loss}

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
		
		##################################################################################################################
		# OBJECTS ANOMALY SCORES
		class_objs = [MVTec_DataModule.id2c[i] for i in batch['class_obj'].tolist()] # class objects list within the batch (dim=batch_size)
		class_counter = Counter(class_objs)
		for c in list(class_counter):
			index_list = []
			for i,obj in enumerate(class_objs):
				if obj==c:
					index_list.append(i)
			anomaly_sum = (np.take(anomaly_scores.detach().cpu().numpy(), np.array(index_list)).sum()) / class_counter[c]	
			self.log("anomaly."+c, anomaly_sum, on_step=False, on_epoch=True, prog_bar=False)
        ##################################################################################################################
   
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
		self.log("auroc", self.val_auroc(pred, batch['label']), on_step=False, on_epoch=True, prog_bar=True, batch_size=imgs.shape[0])
		# IMAGES
		images = self.get_images_for_log(imgs[0:self.hparams.log_images], recon_imgs[0:self.hparams.log_images])
		return {"images": images}

	def validation_epoch_end(self, outputs):
		if self.global_step%self.hparams.log_image_each_epoch==0:
			# we randomly select one batch index
			bidx = random.randrange(100) % len(outputs)
			images = outputs[bidx]["images"]
			self.logger.experiment.log({f"images": images})