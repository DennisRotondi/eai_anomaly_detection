import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .AE_simple import AE

class CODE_AE(AE):
	def __init__(self, hparams):
		super(CODE_AE, self).__init__(hparams)

	def forward(self, img):
		# this implement the denoising autoencoder mechanism
		# here we preferred random whitenoise over zero-noise 
		if self.hparams.noise > 0:
			img = img + torch.rand_like(img)*self.hparams.noise
			# we need to fix the values in the range [-1,1]
			img = img.clamp(min=-1, max=1)
		return self.decoder(self.encoder(img))

	def configure_optimizers(self):
		# note wd can be != 0, this implement the contractive autoencoder behaviour
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