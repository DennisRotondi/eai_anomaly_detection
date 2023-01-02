from dataclasses import dataclass

@dataclass
class Hparams:
    # dataloader params
    dataset_dir: str = "dataset/mvtec_anomaly_detection"
    img_size: int = 256  # size of image
    batch_size: int = 256  # size of the batches
    n_cpu: int = 8  # number of cpu threads to use for the dataloaders
    pin_memory: bool = False # parameter to pin memory in dataloader
    # autoencoder params
    latent_size: int = 128 # size of autoencoder latent space
    lr: float = 1e-3
    log_images: int = 4 # how many images to log each time
    threshold: float = 0
    t_weight: float = 0.7 # how much weight the new threshold wrt the old
    loss_weight: float = 100 # how much weight the reconstruction loss between two pixels 
    min_lr: float = 1e-8 # min lr for ReduceLROnPlateau
    adam_eps: float = 1e-6 #  term added to the denominator to improve numerical stability
    w_std: int = +1 # this param could be 1 (sum std) -1 (sub std)
    wd: float = 1e-6 # weight decay for the contractive strategy
    noise: float = 0.1 # noise factor in the image for denoising strategy