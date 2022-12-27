from dataclasses import dataclass

@dataclass
class Hparams:
    img_size: int = 256  # size of image
    batch_size: int = 256  # size of the batches
    n_cpu: int = 8  # number of cpu threads to use for the dataloaders
    dataset_dir: str = "dataset/mvtec_anomaly_detection"
    lr: float = 1e-4
    latent_size: int = 128 # size of autoencoder latent space
    log_images: int = 4 #how many images to log each time
    # to add wd, betas, ...