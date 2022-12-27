import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.utils
import wandb
import pytorch_lightning as pl
from .data_module import MVTec_DataModule
import random
from torchmetrics import Accuracy

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
    """ Simple Autoencoder """
    def __init__(self, hparams):
        super(AE, self).__init__()
        self.save_hyperparameters(hparams)
        self.encoder = Encoder(self.hparams.latent_size)
        self.decoder = Decoder(self.hparams.latent_size)
        self.threshold = self.hparams.threshold
        self.val_acc = Accuracy()

    def forward(self, img):
        return self.decoder(self.encoder(img))

    def anomaly_score_prediction(self, img, recon):
        recon = recon.view(recon.shape[0],-1)
        img = img.view(img.shape[0],-1)
        anomaly_score = torch.abs(recon-img).mean(dim=-1)
        # anomaly_score = F.normalize(anomaly_score,p=2.0, dim=-1)
        ris = (anomaly_score > self.threshold*self.hparams.threshold_weight).long()
        return ris, anomaly_score.max()

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
        return {"loss": F.mse_loss(100*recon_x, 100*x, reduction='mean')}

    def training_step(self, batch, batch_idx):
        imgs = batch['img']
        recon = self(imgs)
        loss = self.loss_function(recon, imgs)
        self.log_dict(loss)
        _, a = self.anomaly_score_prediction(imgs,recon)
        return {'loss': loss['loss'], 'anom': a}

    def training_epoch_end(self, outputs):
        a = torch.stack([x['anom'] for x in outputs])
        a_avg = a.mean()
        a_max = a.max()
        # here we update the threshold
        self.threshold = a_max
        self.log("anomaly_threshold", a_max, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict({"avg_anomaly_score": a_avg, 'max_anomaly_score': a_max})

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
        # loss = self.loss_function(recon_imgs, imgs)
        pred, _ = self.anomaly_score_prediction(imgs,recon_imgs)
        self.val_acc(pred, batch['label'])
        self.log("accuracy", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        images = self.get_images_for_log(imgs[0:self.hparams.log_images], recon_imgs[0:self.hparams.log_images])
        return {"images": images}

    def validation_epoch_end(self, outputs):
        """ Implements the behaviouir at the end of a validation epoch
        Currently it gathers all the produced examples and log them to wandb,
        limiting the logged examples to `hparams["log_images"]`.
        Then computes the mean of the losses and returns it. 
        Updates the progress bar label with this loss.
        :param outputs: a sequence that aggregates all the outputs of the validation steps
        :returns: the aggregated validation loss and information to update the progress bar
        """
        # we pick one random sample to log
        bidx = random.randrange(100) % len(outputs)
        images = outputs[bidx]["images"]
        self.logger.experiment.log(
            {f"images": images}
            # step=self.global_step,
        )
        # avg_loss = torch.stack([x["loss_val"] for x in outputs]).mean()
        # self.log_dict({"avg_val_loss": avg_loss})
        # return {"avg_val_loss": avg_loss}