from FDDM import *
from omegaconf import OmegaConf

if __name__ == '__main__':

    config_path = "FDDM.yaml"
    config = OmegaConf.load(config_path)  # Loading configuration files
    model = FDDM(**config.get("model").get("params", dict()))

    # Training the model
    model.train_step()

