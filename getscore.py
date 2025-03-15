from FDDM import *
from omegaconf import OmegaConf
from score.fid import calculate_fid_given_paths
from score.clip import *
from score.Is import *
from transformers import CLIPProcessor, CLIPModel


if __name__ == '__main__':
    config_path = "FDDM.yaml"
    config = OmegaConf.load(config_path)  # 加载配置文件
    model = FDDM(**config.get("model").get("params", dict()))

    fake_directory = f"/root/autodl-tmp/score_img"  # 替换为你需要的图像生成路径
    txt_path_fake = f"/root/autodl-tmp/score_txt"  # 替换为你的文字保存路径

    # FID

    real_directory = model.image_directory
    model_path = "FID/pt_inception-2015-12-05-6726825d.pth"  # 权重文件路径

    fid_value = calculate_fid_given_paths(
        [real_directory, fake_directory],  # 两个路径列表
        model_path=model_path,
        batch_size=64,  # 批量大小
        device=torch.device("cuda"),  # 设备
        dims=2048  # 特征维度,必须为2048
    )
    print(f"FID Score: {fid_value:.4f}")
    with open("score.txt", "a") as file:
        file.write(f"FID Score: {fid_value:.4f}\n")

    # CLIP

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16

    clip_path = model.CLIP_path
    img_path_fake = fake_directory

    clip_model = CLIPModel.from_pretrained(clip_path).to(device).eval()  # 加载CLIP模型和处理器（包含图像&文本编码器）
    tokenizer = CLIPTokenizer.from_pretrained(clip_path)  # 分词器
    transformer = CLIPTextModel.from_pretrained(clip_path).to(device).eval()  # 将文本编码为高维向量表示
    processor = CLIPProcessor.from_pretrained(clip_path)

    transform = Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor()  # 将图像转换为张量
        # transforms.Lambda(lambda t: (t * 2) - 1)  # 将张量的值从 [0, 1] 映射到 [-1, 1]
    ])

    dict_text_fake = get_d1ct_text(txt_path_fake)
    dataset_fake = txt3img_Dataset(img_path_fake, dict_text_fake, transform=transform, processor=processor)
    dataloader_fake = DataLoader(dataset_fake, batch_size=batch_size, shuffle=False)
    # 计算生成数据CLIP Score
    fake_mean, fake_std = compute_clip_score(clip_model, dataloader_fake, device=device)

    print(f"Generated Data CLIP Score: Mean={fake_mean:.4f}, Std={fake_std:.4f}")
    with open("score.txt", "a") as file:
        file.write(f"Generated Data CLIP Score: Mean={fake_mean:.4f}, Std={fake_std:.4f}\n")

    # IS

    device = "cuda"
    real_directory = '/root/data/train/flower/img/jpg'
    model_path = "inception_v3_google-0cc3c7bd.pth"  # 替换为你的权重文件路径
    model = get_model(model_path=model_path, device="cuda")

    path = fake_directory
    dataloader = get_dataloader(path=path, batch_size=64, num_workers=1)
    # 计算类别概率
    probs = get_probabilities(model=model, dataloader=dataloader, device=device)
    # 计算IS
    is_score = inception_score(probs, num_splits=100)
    print(f"Inception Score: {is_score:.4f}")
    with open("score.txt", "a") as file:
        file.write(f"Inception Score: {is_score:.4f}\n")
