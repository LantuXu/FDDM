from FDDM import *
from omegaconf import OmegaConf
from score.fid import calculate_fid_given_paths
from score.clip import *
from score.Is import *
from transformers import CLIPProcessor, CLIPModel


if __name__ == '__main__':
    config_path = "FDDM.yaml"
    config = OmegaConf.load(config_path)  # Loading configuration files
    model = FDDM(**config.get("model").get("params", dict()))

    fake_directory = f"/root/autodl-tmp/score_img"  # paths for the Generated images
    txt_path_fake = f"/root/autodl-tmp/score_txt"  # Path for saved the texts

    # FID

    real_directory = model.image_directory
    model_path = "google/inception_v3"  # Weight file path

    fid_value = calculate_fid_given_paths(
        [real_directory, fake_directory],  # List of two paths
        model_path=model_path,
        batch_size=64,
        device=torch.device("cuda"),
        dims=2048  # Feature dimension, must be 2048
    )
    print(f"FID Score: {fid_value:.4f}")
    with open("score.txt", "a") as file:
        file.write(f"FID Score: {fid_value:.4f}\n")

    # CLIP

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16

    clip_path = model.CLIP_path
    img_path_fake = fake_directory

    clip_model = CLIPModel.from_pretrained(clip_path).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(clip_path)
    transformer = CLIPTextModel.from_pretrained(clip_path).to(device).eval()
    processor = CLIPProcessor.from_pretrained(clip_path)

    transform = Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dict_text_fake = get_d1ct_text(txt_path_fake)
    dataset_fake = txt3img_Dataset(img_path_fake, dict_text_fake, transform=transform, processor=processor)
    dataloader_fake = DataLoader(dataset_fake, batch_size=batch_size, shuffle=False)
    # Calculate the generated data CLIP Score
    fake_mean, fake_std = compute_clip_score(clip_model, dataloader_fake, device=device)

    print(f"Generated Data CLIP Score: Mean={fake_mean:.4f}, Std={fake_std:.4f}")
    with open("score.txt", "a") as file:
        file.write(f"Generated Data CLIP Score: Mean={fake_mean:.4f}, Std={fake_std:.4f}\n")

    # IS

    device = "cuda"
    real_directory = '/root/data/train/flower/img/jpg'
    model_path = "google/inception_v3"  # Replace with your weight file path
    model = get_model(model_path=model_path, device="cuda")

    path = fake_directory
    dataloader = get_dataloader(path=path, batch_size=64, num_workers=1)
    # Calculating class probabilities
    probs = get_probabilities(model=model, dataloader=dataloader, device=device)
    # Calculating IS
    is_score = inception_score(probs, num_splits=100)
    print(f"Inception Score: {is_score:.4f}")
    with open("score.txt", "a") as file:
        file.write(f"Inception Score: {is_score:.4f}\n")
