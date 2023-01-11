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

def conv_block(in_features, out_features, kernel_size, stride, padding, bias, slope, normalize = True):
	layer = [nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
	if normalize:
		layer += [nn.BatchNorm2d(out_features)]
	if slope > 0:
		layer += [nn.LeakyReLU(slope, inplace = True)]
	elif slope==0:
		layer += [nn.ReLU(inplace = True)]
	return layer

def deconv_block(in_features, out_features, kernel_size, stride, padding, bias, slope, normalize = True):
	layer = [nn.ConvTranspose2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
	if normalize:
		layer += [nn.BatchNorm2d(out_features)]
	if slope > 0:
		layer += [nn.LeakyReLU(slope, inplace = True)]
	elif slope==0:
		layer += [nn.ReLU(inplace = True)]
	return layer

class Encoder(nn.Module):
	def __init__(self, latent_dim, channels, hparams):
		super().__init__()
		""" the input image is 3x256x256, this size allow to preserve the high res + still load all the dataset"""
		self.convolutions = nn.Sequential(
			*conv_block(channels, 16, kernel_size=4, stride=2, padding=1, bias=True, slope = hparams.slope, normalize=True),
			*[m for mod in [conv_block(2**i, 2**(i+1), kernel_size=4, stride=2, padding=1, bias=True, slope = hparams.slope, normalize=True) \
				for i in range(4,9)] for m in mod],
			*conv_block(512, 1024, kernel_size=4, stride=1, padding=0, bias=True, slope = -1 , normalize=False)
		)
		self.fc = nn.Sequential(
			nn.Linear(1024,512),
			nn.ReLU(inplace=True),
			nn.Linear(512,256),
			nn.ReLU(inplace=True),
			nn.Linear(256, latent_dim),
			nn.ReLU(inplace=True)
		)

	def forward(self, img):
		return self.fc(self.convolutions(img).squeeze(-1).squeeze(-1))

class Decoder(nn.Module):
	def __init__(self, latent_dim, channels, hparams):
		super().__init__()
		self.deconvolution = nn.Sequential(
			*deconv_block(latent_dim, 1024, kernel_size=4, stride=1, padding=0, bias=True, slope = hparams.slope, normalize=True),
			*[m for mod in [deconv_block(2**(i+1), 2**i, kernel_size=4, stride=2, padding=1, bias=True, slope = hparams.slope, normalize=True) \
				for i in range(9,4,-1)] for m in mod],
			*deconv_block(32, channels, kernel_size=4, stride=2, padding=1, bias=True, slope = -1, normalize=False))

	def forward(self, input):
		return torch.tanh(self.deconvolution(input.unsqueeze(-1).unsqueeze(-1)))

# here an alternative to these double architecture using a UNET variant.
# UP UNET module taken from official implementaton, padding strategy to make the solution work
# https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
# https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
class Up(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.up = nn.Sequential(
			*deconv_block(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=True, slope = -1, normalize=False),
		)
		self.conv = nn.Sequential(
			*conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True),
			*conv_block(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True),
		)
	def forward(self, x1, x2):
		x1 = self.up(x1)
		# input is CHW
		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]
		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
						diffY // 2, diffY - diffY // 2])
		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)

class EncoderDecoder_tmp(nn.Module):
	def __init__(self, channels, hparams):
		super().__init__()
		""" the input image for this version is 3x224x224"""
		self.convolutions = nn.Sequential(
			nn.Sequential(
				*conv_block(channels, 64, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True),
				*conv_block(64, 64, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True)),
			nn.Sequential(
				nn.MaxPool2d(2),
				*conv_block(64, 128, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True),
				*conv_block(128, 128, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True)),
			nn.Sequential(
				nn.MaxPool2d(2),
				*conv_block(128, 256, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True),
				*conv_block(256, 256, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True)),
			nn.Sequential(
				nn.MaxPool2d(2),
				*conv_block(256, 512, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True),
				*conv_block(512, 512, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True)),
			nn.Sequential(
				nn.MaxPool2d(2),
				*conv_block(512, 1024, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True),
				*conv_block(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True))
		)
		self.ups = nn.Sequential(
			Up(1024,512),
			Up(512,256),
			Up(256,128),
			# Up(128,64)
			)
		self.final_rec = nn.Sequential(
			*deconv_block(128, 64, kernel_size=2, stride=2, padding=0, bias=True, slope = -1, normalize=False),
			*conv_block(64, channels, kernel_size=3, stride=1, padding="same", bias=False, slope = 0, normalize=True),
			*conv_block(channels, channels, kernel_size=3, stride=1, padding="same", bias=False, slope = -1, normalize=True))
	
	def forward(self, img, latent = False):
		# x1 = self.convolutions[0](img)
		x2 = self.convolutions[0:2](img)
		x3 = self.convolutions[2](x2)
		x4 = self.convolutions[3](x3)
		x5 = self.convolutions[4](x4)
		x = self.ups[0](x5, x4)
		x = self.ups[1](x, x3)
		x = self.ups[2](x, x2)
		# note, to reduce the power of the network here we don't see the first features extracted (x1)
		if latent:
			return x5, torch.tanh(self.final_rec(x))
		else:
			return torch.tanh(self.final_rec(x))

# idea to skip features at level 1,2
class EncoderDecoder_tmp2(nn.Module):
	def __init__(self, channels, hparams):
		super().__init__()
		""" the input image for this version is 3x224x224"""
		self.convolutions = nn.Sequential(
			nn.Sequential(
				*conv_block(channels, 64, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True),
				*conv_block(64, 64, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True)),
			nn.Sequential(
				nn.MaxPool2d(2),
				*conv_block(64, 128, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True),
				*conv_block(128, 128, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True)),
			nn.Sequential(
				nn.MaxPool2d(2),
				*conv_block(128, 256, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True),
				*conv_block(256, 256, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True)),
			nn.Sequential(
				nn.MaxPool2d(2),
				*conv_block(256, 512, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True),
				*conv_block(512, 512, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True)),
			nn.Sequential(
				nn.MaxPool2d(2),
				*conv_block(512, 1024, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True),
				*conv_block(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True))
		)
		self.ups = nn.Sequential(
			Up(1024,512),
			Up(512,256),
			# Up(256,128),
			# Up(128,64)
			)
		self.final_rec = nn.Sequential(
			*deconv_block(256, 128, kernel_size=2, stride=2, padding=0, bias=True, slope = -1, normalize=False),
			*conv_block(128, 128, kernel_size=3, stride=1, padding="same", bias=False, slope = 0, normalize=True),
			*conv_block(128, 128, kernel_size=3, stride=1, padding="same", bias=False, slope = 0, normalize=True),
			*deconv_block(128, 64, kernel_size=2, stride=2, padding=0, bias=True, slope = -1, normalize=False),
			*conv_block(64, channels, kernel_size=3, stride=1, padding="same", bias=False, slope = 0, normalize=True),
			*conv_block(channels, channels, kernel_size=3, stride=1, padding="same", bias=False, slope = -1, normalize=True))
	
	def forward(self, img, latent = False):
		# x1 = self.convolutions[0](img)
		# x2 = self.convolutions[0:2](img)
		x3 = self.convolutions[0:3](img)
		x4 = self.convolutions[3](x3)
		x5 = self.convolutions[4](x4)
		x = self.ups[0](x5, x4)
		x = self.ups[1](x, x3)
		#x = self.ups[2](x, x2)
		# note, to reduce the power of the network here we don't see the first features extracted (x1,x2)
		if latent:
			return x5, torch.tanh(self.final_rec(x))
		else:
			return torch.tanh(self.final_rec(x))

#idea to remove U blocks	
class EncoderDecoder(nn.Module):
	def __init__(self, channels, hparams):
		super().__init__()
		""" the input image for this version is 3x224x224"""
		self.encoder = nn.Sequential(
			nn.Sequential(
				*conv_block(channels, 64, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True),
				*conv_block(64, 64, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True)),
			nn.Sequential(
				nn.MaxPool2d(2),
				*conv_block(64, 128, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True),
				*conv_block(128, 128, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True)),
			nn.Sequential(
				nn.MaxPool2d(2),
				*conv_block(128, 256, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True),
				*conv_block(256, 256, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True)),
			nn.Sequential(
				nn.MaxPool2d(2),
				*conv_block(256, 512, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True),
				*conv_block(512, 512, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True)),
			nn.Sequential(
				nn.MaxPool2d(2),
				*conv_block(512, 1024, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True),
				*conv_block(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True))
		)
		
		self.decoder = nn.Sequential(
			*deconv_block(1024, 1024, kernel_size=2, stride=2, padding=0, bias=True, slope = -1, normalize=False),
			*conv_block(1024, 512, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True),
			*conv_block(512, 512, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True),
			
            *deconv_block(512, 512, kernel_size=2, stride=2, padding=0, bias=True, slope = -1, normalize=False),
			*conv_block(512, 256, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True),
			*conv_block(256, 256, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True),

            *deconv_block(256, 256, kernel_size=2, stride=2, padding=0, bias=True, slope = -1, normalize=False),
			*conv_block(256, 128, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True),
			*conv_block(128, 128, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True),

            *deconv_block(128, 128, kernel_size=2, stride=2, padding=0, bias=True, slope = -1, normalize=False),
			*conv_block(128, 64, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True),
			*conv_block(64, 64, kernel_size=3, stride=1, padding=1, bias=False, slope = 0, normalize=True),

			*conv_block(64, 3, kernel_size=3, stride=1, padding=1, bias=False, slope = 0.2, normalize=True),
			*conv_block(3, 3, kernel_size=3, stride=1, padding=1, bias=False, slope = 0.2, normalize=True))
	
	def forward(self, img, latent = False):
		if latent:
			latent = self.encoder(img)
			return latent, torch.tanh(self.decoder(latent))
		else:
			return torch.tanh(self.decoder(self.encoder(img)))


class AE(pl.LightningModule):
	""" Simple Autoencoder """
	def __init__(self, hparams):
		super(AE, self).__init__()
		self.save_hyperparameters(hparams)
		if self.hparams.version == "1":
			self.encoder = Encoder(self.hparams.latent_size, self.hparams.img_channels, self.hparams)
			self.decoder = Decoder(self.hparams.latent_size, self.hparams.img_channels, self.hparams)
			if self.hparams.gaussian_initialization:
				# https://pytorch.org/docs/master/generated/torch.nn.Module.html?highlight=apply#torch.nn.Module.apply
				self.encoder.apply(self.weights_init_normal) # apply(f) applies 'f' to all the submodules of the network
				self.decoder.apply(self.weights_init_normal)
		elif self.hparams.version == "2":
			self.encoderdecoder = EncoderDecoder(self.hparams.img_channels, self.hparams)
			if self.hparams.gaussian_initialization:
				self.encoderdecoder.apply(self.weights_init_normal) # apply(f) applies 'f' to all the submodules of the network

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
		if self.hparams.version == "1":
			return self.decoder(self.encoder(img))
		elif self.hparams.version == "2":
			return self.encoderdecoder(img)
	
	def anomaly_score(self, img, recon): # (batch, 3, 256, 256)	
		if self.hparams.anomaly_strategy == "mse":
			#The maximum anomaly score with MSE that two images 
			#can obtain in our setting is 2x3x256x256 = 393216.
			#We could even normalize the result by dividing it by this value!
			recon = recon.view(recon.shape[0],-1) # a sort of flatten operation
			img = img.view(img.shape[0],-1)
			return (torch.abs(recon-img).sum(-1)) 
		elif self.hparams.anomaly_strategy == "ssim":
			# also in this case to apply the treshold mechanism we compute the dissimilarity
			return (1-SSIM(recon, img, data_range=2.0, k1=0.01, k2=0.03, reduction=None))/2
		else:
			return (1-MSSIM(recon, img, data_range=2.0, k1=0.01, k2=0.03, reduction=None))/2
	
	def anomaly_prediction(self, img, recon=None, batch = None):
		if recon is None:
			recon = self(img)
		anomaly_score = self.anomaly_score(img, recon)
		ris = (anomaly_score > self.hparams.threshold).long()
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

	def main_loss(self, recon_x, x):
		if self.hparams.training_strategy == "mse":
			loss = torch.nn.functional.mse_loss(recon_x, x, reduction=self.hparams.reduction)
		elif self.hparams.training_strategy == "ssim":
			# note that we want to maximize SSIM, this is equivalent to minimize Structural Dissimilarity DSSIM: (1-SSIM)/2
			loss = (1-SSIM(recon_x, x, data_range=2.0, k1=0.01, k2=0.03))/2
		else:
			loss = (1-MSSIM(recon_x, x, data_range=2.0, k1=0.01, k2=0.03))/2
		# the loss_weight hparams module the importance of the main loss
		return self.hparams.loss_weight*loss

	def loss_function(self,recon_x, x):
		# the loss function is simply the main loss here
		return {"loss": self.main_loss(recon_x, x)}

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
		a_std = anomaly_scores.std().detach().cpu()
		# a_mean = anomaly_scores.detach().cpu().numpy()
		# a_std = anomaly_scores.std().detach().cpu().numpy()
		
		##################################################################################################################
		# OBJECTS ANOMALY SCORES
		class_objs = [MVTec_DataModule.id2c[i] for i in batch['class_obj'].tolist()] # class objects list within the batch (dim=batch_size)
		class_counter = Counter(class_objs)
		for c in list(class_counter):
			index_list = [i for i,obj in enumerate(class_objs) if obj==c]
			anomaly_sum = (np.take(anomaly_scores.detach().cpu().numpy(), np.array(index_list)).sum()) / class_counter[c]	
			self.log("anomaly_score."+c, anomaly_sum, on_step=False, on_epoch=True, prog_bar=False)
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
		self.hparams.threshold = (1-self.hparams.t_weight)*self.hparams.threshold + \
							self.hparams.t_weight*(self.avg_anomaly + self.hparams.w_std*self.std_anomaly)
		self.log("anomaly_threshold", self.hparams.threshold, on_step=False, on_epoch=True, prog_bar=True)

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
		# good practice https://github.com/Lightning-AI/lightning/issues/4396
		self.val_precision.update(pred, batch['label'])
		self.val_recall.update(pred, batch['label'])
		self.val_f1score.update(pred, batch['label'])
		self.val_auroc.update(pred, batch['label'])
		self.log("precision", self.val_precision, on_step=False, on_epoch=True, prog_bar=True, batch_size=imgs.shape[0])
		self.log("recall", self.val_recall, on_step=False, on_epoch=True, prog_bar=True, batch_size=imgs.shape[0])
		self.log("f1_score", self.val_f1score, on_step=False, on_epoch=True, prog_bar=True, batch_size=imgs.shape[0])
		self.log("auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True, batch_size=imgs.shape[0])
		# IMAGES
		images = self.get_images_for_log(imgs[0:self.hparams.log_images], recon_imgs[0:self.hparams.log_images])
		return {"images": images}

	def validation_epoch_end(self, outputs):
		if self.global_step%self.hparams.log_image_each_epoch==0:
			# we randomly select one batch index
			bidx = random.randrange(100) % len(outputs)
			images = outputs[bidx]["images"]
			self.logger.experiment.log({f"images": images})