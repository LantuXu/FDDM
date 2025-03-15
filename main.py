from FDDM import *
from omegaconf import OmegaConf

if __name__ == '__main__':

    config_path = "FDDM.yaml"
    config = OmegaConf.load(config_path)  # 加载配置文件
    model = FDDM(**config.get("model").get("params", dict()))

    # 训练模型
    model.train_step()

