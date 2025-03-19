import torch
import torch.fft
from gaussian_mask import gaussian_mask

class FFTtransform():

    def ffttrans(self, image):
        dft = torch.fft.fft2(image)
        # Separate the real and imaginary parts
        real_part = dft.real
        imag_part = dft.imag

        # Concatenate the real and imaginary parts
        FFT = torch.cat((real_part, imag_part), dim=1)  # Shape: (B, 2 * C, H, W)
        return FFT

    def low_pass_filter(self, image, radius=30, gaussian=False):
        dft = torch.fft.fft2(image)
        dft_shift = torch.fft.fftshift(dft)  # Shift the low frequencies to the center
        B, C, H, W = image.shape
        if gaussian:
            mask = gaussian_mask(img_size=H, sigma=1)
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = mask.expand(B, C, H, W)
            mask = mask.to(image.device)
        else:
            y, x = torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32)
            y, x = torch.meshgrid(y, x, indexing='ij')

            center_h, center_w = H // 2, W // 2
            distance = torch.sqrt((x - center_w) ** 2 + (y - center_h) ** 2)

            # create the circular mask
            mask = (distance <= radius).float()
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = mask.expand(B, C, H, W)
            mask = mask.to(image.device)

        dft_filtered = dft_shift * mask

        # Computing inverse FFT
        dft = torch.fft.ifftshift(dft_filtered)

        # Separate the real and imaginary parts
        real_part = dft.real
        imag_part = dft.imag

        # Concatenate the real and imaginary parts
        FFT_filtered = torch.cat((real_part, imag_part), dim=1)  # Shape: (B, 2 * C, H, W)
        return FFT_filtered

    def high_pass_filter(self, image, radius=30, gaussian=False):

        # Compute FFT
        dft = torch.fft.fft2(image)

        # Create the high-pass mask
        B, C, H, W = image.shape
        if gaussian:
            mask = gaussian_mask(img_size=H, sigma=1)
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = mask.expand(B, C, H, W)
            mask = mask.to(image.device)
        else:
            y, x = torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32)
            y, x = torch.meshgrid(y, x, indexing='ij')

            center_h, center_w = H // 2, W // 2
            distance = torch.sqrt((x - center_w) ** 2 + (y - center_h) ** 2)

            # create the circular mask
            mask = (distance <= radius).float()
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = mask.expand(B, C, H, W)
            mask = mask.to(image.device)

        dft = dft * mask

        # Separate the real and imaginary parts
        real_part = dft.real
        imag_part = dft.imag

        # Concatenate the real and imaginary parts
        FFT_filtered = torch.cat((real_part, imag_part), dim=1)  # Shape: (B, 2 * C, H, W)
        return FFT_filtered