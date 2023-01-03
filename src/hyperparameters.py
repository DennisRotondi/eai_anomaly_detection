from dataclasses import dataclass

@dataclass
class Hparams:
    # dataloader params
    #dataset_dir: str = "../data_v1" # LEO
    dataset_dir: str = "dataset/mvtec_anomaly_detection" # DENNIS
    img_size: int = 256  # size of image
    img_channels: int = 3
    batch_size: int = 256  # size of the batches
    n_cpu: int = 8  # number of cpu threads to use for the dataloaders
    pin_memory: bool = False # parameter to pin memory in dataloader
    # autoencoder params
    latent_size: int = 512 # size of autoencoder latent space
    lr: float = 1e-3
    log_images: int = 4 # how many images to log each time
    threshold: float = 0
    t_weight: float = 0.8 # how much weight the new threshold wrt the old
    loss_weight: float = 10 # how much weight the reconstruction loss between two pixels 
    min_lr: float = 1e-8 # min lr for ReduceLROnPlateau
    adam_eps: float = 1e-6 #  term added to the denominator to improve numerical stability
    w_std: int = +1 # this param could be 1 (sum std) -1 (sub std)
    wd: float = 1e-6 # weight decay for the contractive strategy
    noise: float = 0.1 # noise factor in the image for denoising strategy
    # logging params
    log_image_each_epoch: int = 2 # epochs interval we wait to log images   