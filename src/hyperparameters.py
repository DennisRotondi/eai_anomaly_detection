from dataclasses import dataclass

@dataclass
class Hparams:
    # dataloader params
    dataset_dir: str = "dataset/mvtec_anomaly_detection" # DENNIS
    img_size: int = 256  # size of image
    img_channels: int = 3
    batch_size: int = 256  # size of the batches
    n_cpu: int = 8  # number of cpu threads to use for the dataloaders
    pin_memory: bool = False # parameter to pin memory in dataloader
    # autoencoder params
    latent_size: int = 128 # from the mvtec paper 128 # size of autoencoder latent space
    lr: float = 1e-3
    threshold: float = 0.5
    gaussian_initialization: bool = True # perform or not the Gaussian inizialization
    augmentation: bool = False # apply augmentation startegy to input images
    obj_classes: int = 15 # number of classes in the dataset
    t_weight: float = 0.7 # how much weight the new threshold wrt the old
    loss_weight: float = 1 # how much weight the reconstruction loss between two pixels 
    min_lr: float = 1e-8 # min lr for ReduceLROnPlateau
    adam_eps: float = 1e-6 #  term added to the denominator to improve numerical stability
    w_std: float = -0.6 # this param weights how much we are going to add of the std in treshold update
    wd: float = 1e-6 # weight decay as regulation strategy
    noise: float = 0.1 # noise factor in the image for denoising strategy
    contractive: bool = True # choose if apply contraction to the loss of not
    lamb: float = 1e-3 # controls the relative importance of the Jacobian (contractive) loss.
    reduction: str = "mean" # "mean" or "sum" according to the reduction loss strategy
    slope: float = 0.2 # slope for the leaky relu in convolutions
    # mixer stuff
    mixer_ae: bool = True # if you want to *treshold* with the mixer strategy or not
    dropout: float = 0.2 # dropout for the mixer classifier
    cross_w: float = 10 # the importance to give to the classification task wrt reconstruction one
    anomaly_strategy: str = "mssim" # "mssim", "mse", "ssim"
    training_strategy: str = "mssim" # "mssim", "mse", "ssim"
    # logging params
    log_images: int = 4 # how many images to log each time
    log_image_each_epoch: int = 3 # epochs interval we wait to log images   