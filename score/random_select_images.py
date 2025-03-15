import shutil
import os.path
import random


def random_select_images(source_folder, target_folder, n):
    """
    从源文件夹及其子文件夹中随机选取 n 张图片，并将它们复制到目标文件夹中。

    :param source_folder: 源文件夹路径
    :param target_folder: 目标文件夹路径
    :param n: 需要随机选取的图片数量
    """
    # 获取源文件夹及其子文件夹中的所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    image_files = []
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_files.append(os.path.join(root, file))

    # 检查源文件夹中是否有图片
    if not image_files:
        print(f"No images found in {source_folder}.")
        return

    # 检查 n 是否大于图片总数
    if n > len(image_files):
        print(f"Warning: Only {len(image_files)} images found. Selecting all images.")
        n = len(image_files)

    # 随机选取 n 张图片
    selected_images = random.sample(image_files, n)

    # 创建目标文件夹（如果不存在）
    os.makedirs(target_folder, exist_ok=True)

    # 复制选中的图片到目标文件夹
    for image in selected_images:
        target_path = os.path.join(target_folder, os.path.basename(image))
        shutil.copy2(image, target_path)
