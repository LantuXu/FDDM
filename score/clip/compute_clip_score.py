from tqdm import tqdm
import torch
import numpy as np

def compute_clip_score(model, dataloader, device="cuda"):
    scores = []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), desc='Calculating CLIP Score', unit='it',
                                                 total=len(dataloader)):

            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # 获取图像和文本特征
            image_features = model.get_image_features(pixel_values)
            text_features = model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # 计算余弦相似度
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features * text_features).sum(dim=-1)
            scores.extend(similarity.cpu().numpy())

    return np.mean(scores), np.std(scores)