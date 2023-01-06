import torch
from .AE_simple import AE
from torch import nn
import torch.nn.functional as F
from .data_module import MVTec_DataModule
from torchmetrics import F1Score
import numpy as np

class Obj_classifer(nn.Module):
	def __init__(self, latent_dim, out_classes, hparams):
		super().__init__()
		self.classifier = nn.Sequential(
			nn.Linear(latent_dim,latent_dim//2),
			nn.ReLU(inplace=True),
			nn.Dropout(hparams.dropout),
			nn.Linear(latent_dim//2, out_classes),
			nn.ReLU(inplace=True)
		)
	def forward(self, latent):
		return self.classifier(latent)

class Mixer_AE(AE):
	def __init__(self, hparams):
		super(Mixer_AE, self).__init__(hparams)
		self.classifier = Obj_classifer(self.hparams.latent_size, self.hparams.obj_classes, self.hparams)
		# if you want to predict using different tresholds you need to store 
		# different tresholds. If you prefer not then you can average all of them.
		self.thresholds = {a: self.threshold for a in MVTec_DataModule.id2c.keys()}
		## metric to log the classification problem
		self.val_f1score_classes = F1Score(task = 'multiclass', num_classes = self.hparams.obj_classes, average = 'macro')
	# in what follow we implement optionally the Contractive and Denoising behaviour	
	def forward(self, img):
		if self.hparams.noise > 0:
			img = img + torch.rand_like(img)*self.hparams.noise
			img = img.clamp(min=-1, max=1)
		latent = self.encoder(img)
		return self.decoder(latent), self.classifier(latent)
	
	def anomaly_prediction(self, img, recon = None, classes = None):
		if recon is None:
			recon, classes = self(img)
		anomaly_score = self.anomaly_score(img, recon)
		if self.hparams.mixer_ae:
			classes = torch.argmax(classes,dim=-1).tolist()
			idx = [self.thresholds[i] for i in classes]
			ris = (anomaly_score > torch.tensor(idx, device = self.device)).long()
		else:
			ris = (anomaly_score > self.threshold).long()
		return ris
	
	def loss_function(self,recon_x, x, classes):	
		loss = self.hparams.loss_weight*(recon_x-x['img'])**2
		if self.hparams.contractive:
			weights = torch.concat([param.view(-1) for param in self.encoder.parameters()])
			jacobian_loss = self.hparams.lamb*weights.norm(p='fro')
			loss = loss + jacobian_loss
		# here we compute the cross entropy prodiction for the classes
		CE = self.hparams.cross_w*F.cross_entropy(classes, x["class_obj"])

		loss = loss +  CE
		if self.hparams.reduction == 'mean':
			loss = loss.mean()
		else:
			loss = loss.sum()
		return {"loss": loss, "CE":CE}

	def training_step(self, batch, batch_idx):
		imgs = batch['img']
		recon, classes = self(imgs)
		loss = self.loss_function(recon, batch, classes)
		# LOSS
		self.log_dict(loss)
		# ANOMALY SCORE --> mean and standard deviation for each class
		anomaly_scores = self.anomaly_score(imgs, recon)
		# print(anomaly_scores.shape)
		all_std = dict()
		all_mean = dict()
		for k in self.thresholds.keys():
			all_k = anomaly_scores[batch["class_obj"]==k]
			if all_k.nelement() == 0:
				# we skip if nothing to log
				continue
			all_mean[k] = all_k.mean().detach().cpu().item()
			self.log("anomaly_avg."+MVTec_DataModule.id2c[k], all_mean[k], on_step=False, on_epoch=True, prog_bar=False)
			if all_k.nelement()>1:
				# std with only one element is not defined in pytorch (nan)
				all_std[k] = all_k.std().detach().cpu().item()
				self.log("anomaly_std."+MVTec_DataModule.id2c[k], all_std[k], on_step=False, on_epoch=True, prog_bar=False)
		return {'loss': loss['loss'], 'anom': all_mean, 'a_std': all_std}

	def training_epoch_end(self, outputs):
		# we need to update all the thresholds
		all_tre = list()
		for k in self.thresholds.keys():
			a = np.array([x['anom'][k] for x in outputs if x['anom'].get(k,None) is not None]) 
			a_std = np.array([x['a_std'][k] for x in outputs if x['a_std'].get(k,None) is not None]) 
			avg_anomalyk = a.mean()
			std_anomalyk = a_std.mean()
			# THRESHOLD UPDATE
			self.thresholds[k] = (1-self.hparams.t_weight)*self.thresholds[k] + \
								self.hparams.t_weight*(avg_anomalyk + self.hparams.w_std*std_anomalyk)
			all_tre.append(self.thresholds[k])
			self.log("anomaly_threshold."+MVTec_DataModule.id2c[k], self.thresholds[k], on_step=False, on_epoch=True, prog_bar=False)
		all_tre = np.array(all_tre)
		self.threshold = all_tre.mean()
		self.log("anomaly_threshold_all_avg", self.threshold, on_step=False, on_epoch=True, prog_bar=True)
		
	def validation_step(self, batch, batch_idx):
		imgs = batch['img']
		recon_imgs, classes = self(imgs)
		# LOSS
		self.log("val_loss", self.loss_function(recon_imgs, batch, classes)["loss"], on_step=False, on_epoch=True, batch_size=imgs.shape[0])
		# RECALL, PRECISION, F1 on anomaly predicitons
		pred = self.anomaly_prediction(imgs, recon_imgs, classes=classes)
		self.log("precision", self.val_precision(pred, batch['label']), on_step=False, on_epoch=True, prog_bar=True, batch_size=imgs.shape[0])
		self.log("recall", self.val_recall(pred, batch['label']), on_step=False, on_epoch=True, prog_bar=True, batch_size=imgs.shape[0])
		self.log("f1_score", self.val_f1score(pred, batch['label']), on_step=False, on_epoch=True, prog_bar=True, batch_size=imgs.shape[0])
		self.log("auroc", self.val_auroc(pred, batch['label']), on_step=False, on_epoch=True, prog_bar=True, batch_size=imgs.shape[0])
		# F1 on classes predictions
		classes = torch.argmax(classes, dim = -1)
		self.log("f1_score_classes", self.val_f1score_classes(classes, batch["class_obj"]), on_step=False, on_epoch=True, prog_bar=True, batch_size=imgs.shape[0])
		# IMAGES
		images = self.get_images_for_log(imgs[0:self.hparams.log_images], recon_imgs[0:self.hparams.log_images])
		return {"images": images}