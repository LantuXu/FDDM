import shutil
import os.path
import random


def random_select_images(source_folder, target_folder, n):
    # Get all image files in the source folder and its subfolders
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    image_files = []
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_files.append(os.path.join(root, file))

    if not image_files:
        print(f"No images found in {source_folder}.")
        return

    # Check that n is greater than the total number of images
    if n > len(image_files):
        print(f"Warning: Only {len(image_files)} images found. Selecting all images.")
        n = len(image_files)

    # Select n images at random
    selected_images = random.sample(image_files, n)

    os.makedirs(target_folder, exist_ok=True)

    # Copy the selected image to the destination folder
    for image in selected_images:
        target_path = os.path.join(target_folder, os.path.basename(image))
        shutil.copy2(image, target_path)
