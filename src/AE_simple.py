import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.utils
import wandb
import pytorch_lightning as pl
from typing import Sequence, List, Dict, Tuple, Optional, Any, Set, Union, Callable, Mapping

def conv_block(in_features, out_features, kernel_size, stride, padding, bias, slope, normalize = True, affine = True):
    layer = [nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
    if normalize:
        layer += [nn.BatchNorm2d(out_features, affine=affine)]
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
    """ Variational Autoencoder """
    def __init__(self, hparams):
        super(AE, self).__init__()
        self.save_hyperparameters(hparams)
        self.encoder = Encoder(self.hparams.latent_size)
        self.decoder = Decoder(self.hparams.latent_size)
        # It avoids logging when lighting does a sanity check on the validation
        self.is_sanity = True

    def forward(self, img):
        return self.decoder(self.encoder(img))

    def configure_optimizers(self):
        # note wd = 0 in this simplest version
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=0)
        reduce_lr_on_plateau = ReduceLROnPlateau(optimizer, mode='min',verbose=True, min_lr=1e-8)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr_on_plateau,
                "monitor": 'loss',
                "frequency": 1
            },
        }

    def loss_function(self,recon_x, x):
        """ loss function is mse"""
        return {"loss": F.mse_loss(recon_x, x, reduction='sum')}

    def training_step(self, batch, batch_idx):
        imgs = batch['img']
        loss = self.loss_function(self(imgs), imgs)
        self.log_dict(loss)
        return loss['loss']

    def get_images_for_log(self, real, reconstructed):
        example_images = []
        for i in range(real.shape[0]):
            couple = torchvision.utils.make_grid(
                [real[i], reconstructed[i]],
                nrow=2,
                normalize=True,
                scale_each=False,
                pad_value=1,
                padding=4,
            )
            example_images.append(
                wandb.Image(couple.permute(1, 2, 0).detach().cpu().numpy(), mode="RGB")# no need of .permute(1, 2, 0) since pil image
            )
        return example_images

    def validation_step(self, batch, batch_idx):
        imgs = batch['img']
        recon_imgs = self(imgs)
        loss = self.loss_function(recon_imgs, imgs)
        images = self.get_images_for_log(imgs, recon_imgs)
        return {"loss_vae_val": loss['loss'], "images": images}

    def validation_epoch_end(self, outputs):
        """ Implements the behaviouir at the end of a validation epoch
        Currently it gathers all the produced examples and log them to wandb,
        limiting the logged examples to `hparams["log_images"]`.
        Then computes the mean of the losses and returns it. 
        Updates the progress bar label with this loss.
        :param outputs: a sequence that aggregates all the outputs of the validation steps
        :returns: the aggregated validation loss and information to update the progress bar
        """
        images = []

        for x in outputs:
            images.extend(x["images"])
            
        images = images[: self.hparams.log_images]

        if not self.is_sanity:  # ignore if it not a real validation epoch. The first one is not.
            print(f"Logged {len(images)} images for each category.")
            self.logger.experiment.log(
                {f"images": images},
                step=self.global_step,
            )
        self.is_sanity = False

        avg_loss = torch.stack([x["loss_vae_val"] for x in outputs]).mean()
        self.log_dict({"avg_val_loss_vae": avg_loss})
        return {"avg_val_loss_vae": avg_loss}
