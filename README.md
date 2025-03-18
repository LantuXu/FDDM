# FDDM (Frequency Domain Diffusion Models)

This repository contains the implementation of the Frequency Domain Diffusion Model (FDDM), a deep learning model designed for image generation tasks. 

## Repository Structure

- **main.py**: The main training script. It reads parameters from `FDDM.yaml` and starts the training process.
- **FDDM.yaml**: Configuration file containing training and model parameters.
- **getscore.py**: Script for scoring the generated images. Replace the dataset folder and the training set folder in the YAML file to use this script.
- **generateimg.py**: Script for generating images. It requires checkpoint files and currently has some bugs that require the training set files to be present for image generation. This will be fixed in future updates.

## Configuration Parameters

The `FDDM.yaml` file contains the following parameters:

### Model Parameters

- **time_steps**: Length of time steps, ranging from 1 to 1000.
- **linear_start**: Start value for linear beta.
- **linear_end**: End value for linear beta.
- **base_learning_rate**: Base learning rate for training.
- **batch_size**: Batch size for training.
- **epoch**: Number of epochs for training.
- **image_directory**: Directory containing training images.
- **text_directory**: Directory containing text descriptions for training.
- **img_h**: Height of the images.
- **img_w**: Width of the images.
- **vlb_Switch**: Whether to reconstruct the ELBO loss.
- **v_posterior**: Hyperparameter for interpolating between optimal variance and full forward variance.
- **original_elbo_weight**: Weight for the original ELBO loss.

### Output Parameters

- **snap**: Frequency of saving model parameters.
- **out_path**: Path to save output images.
- **out_num**: Number of images to output.

### Device Parameters

- **device**: Device to use for training (e.g., "cuda").
- **pretrain_device**: Device to use for pretrained models.

### Additional Parameters

- **z_coef**: Coefficient for frequency domain information.
- **z_in**: Number of channels for additional embedded information.
- **z_dim**: Size of the additional embedded information vector.
- **z_layer**: Number of convolutional layers for encoding additional information.
- **mask_type**: Type of mask to apply (1 for high-pass filter, 0 for no filter, -1 for low-pass filter).
- **mask_radius**: Radius of the filter mask.
- **gaussian_mask**: Whether to use a Gaussian mask.
- **gen_txt_path**: Path to the text file used for generation.

### DDIM Parameters

- **DDIM_Switch**: Whether to use DDIM sampling.
- **DDIM_timesteps**: Number of time steps for DDIM sampling.
- **DDIM_eta**: Parameter to adjust the determinism of the generation process.

### VAE Parameters

- **VAE_Switch**: Whether to use VAE.
- **VAE_path**: Path to the VAE model.
- **VAE_config_path**: Path to the VAE configuration file.

### CLIP Parameters

- **CLIP_Switch**: Whether to use CLIP (always enabled, no non-text training interface is set).

### Unet Parameters

- **Unet_config**: Configuration for the Unet model, including channels, attention resolutions, number of residual blocks, and more.

## Usage

1. **Training**: Run `main.py` to start the training process. Ensure that the `FDDM.yaml` file is correctly configured with the desired parameters.

2. **Scoring**: Use `getscore.py` to evaluate the generated images. Modify the dataset and training set folders in the YAML file as needed.

3. **Image Generation**: Use `generateimg.py` to generate images from checkpoint files. Note that this script currently requires the training set files to be present and has some bugs that will be addressed in future updates.
