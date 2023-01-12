import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .AE_CODE import CODE_AE
import pytorch_lightning as pl

class Ensemble_model(pl.LightningModule):
    """_summary_
        Generic class for models, it take as input a list of "Model" and predict using
        a majority voting with uniform weights.
    """
    def __init__(self,models:list, weights):
        self.models=models
        self.num_models=len(models)
        sum_w = sum(weights)
        self.weights = torch.tensor([i/sum_w for i in weights])
    def anomaly_prediction(self, img):
        l = list()
        for m in self.models:
            l.append(m.anomaly_prediction(img))
        preds = torch.stack(l)
        preds = ((preds+self.weights).sum(dim=-1) > self.num_models//2).long()
        return preds