import torch
import torch.fft
from gaussian_mask import gaussian_mask

class FFTtransform():

    def ffttrans(self, image):
        """
        对输入图像进行批量 FFT 变换。
        输入形状: (B, C, H, W)
        输出形状: (B, 2 * C, H, W)
        """
        # 对每个通道进行 FFT
        dft = torch.fft.fft2(image)  # 输出形状: (B, C, H, W)，复数类型

        # 分离实部和虚部
        real_part = dft.real  # 实部，形状: (B, C, H, W)
        imag_part = dft.imag  # 虚部，形状: (B, C, H, W)

        # 拼接实部和虚部
        FFT = torch.cat((real_part, imag_part), dim=1)  # 形状: (B, 2 * C, H, W)
        return FFT

    def low_pass_filter(self, image, radius=30, gaussian=False):
        """
        低通滤波。
        输入形状: (B, C, H, W)
        输出形状: (B, 2 * C, H, W)
        """
        # 计算 FFT
        dft = torch.fft.fft2(image)
        dft_shift = torch.fft.fftshift(dft)  # 将低频移到中心

        # 创建低通掩码
        B, C, H, W = image.shape
        if gaussian:
            mask = gaussian_mask(img_size=H, sigma=1)
            mask = mask.unsqueeze(0).unsqueeze(0)  # 形状: (1, 1, H, W)
            mask = mask.expand(B, C, H, W)  # 形状: (B, C, H, W)
            mask = mask.to(image.device)  # 将掩码移动到与图像相同的设备
        else:
            # 创建网格坐标
            y, x = torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32)
            y, x = torch.meshgrid(y, x, indexing='ij')

            # 计算每个像素到圆心的距离
            center_h, center_w = H // 2, W // 2
            distance = torch.sqrt((x - center_w) ** 2 + (y - center_h) ** 2)

            # 创建圆形掩码
            mask = (distance <= radius).float()  # 圆内为 1，圆外为 0
            mask = mask.unsqueeze(0).unsqueeze(0)  # 形状: (1, 1, H, W)
            mask = mask.expand(B, C, H, W)  # 形状: (B, C, H, W)
            mask = mask.to(image.device)  # 将掩码移动到与图像相同的设备

        # 应用掩码
        dft_filtered = dft_shift * mask

        # 计算逆 FFT
        dft = torch.fft.ifftshift(dft_filtered)

        # 分离实部和虚部
        real_part = dft.real  # 实部，形状: (B, C, H, W)
        imag_part = dft.imag  # 虚部，形状: (B, C, H, W)

        # 拼接实部和虚部
        FFT_filtered = torch.cat((real_part, imag_part), dim=1)  # 形状: (B, 2 * C, H, W)
        return FFT_filtered

    def high_pass_filter(self, image, radius=30, gaussian=False):
        """
        高通滤波。
        输入形状: (B, C, H, W)
        输出形状: (B, 2 * C, H, W)
        """
        # 计算 FFT
        dft = torch.fft.fft2(image)

        # 创建高通掩码
        B, C, H, W = image.shape
        if gaussian:
            mask = gaussian_mask(img_size=H, sigma=1)
            mask = mask.unsqueeze(0).unsqueeze(0)  # 形状: (1, 1, H, W)
            mask = mask.expand(B, C, H, W)  # 形状: (B, C, H, W)
            mask = mask.to(image.device)  # 将掩码移动到与图像相同的设备
        else:
            # 创建网格坐标
            y, x = torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32)
            y, x = torch.meshgrid(y, x, indexing='ij')

            # 计算每个像素到圆心的距离
            center_h, center_w = H // 2, W // 2
            distance = torch.sqrt((x - center_w) ** 2 + (y - center_h) ** 2)

            # 创建圆形掩码
            mask = (distance <= radius).float()  # 圆内为 1，圆外为 0
            mask = mask.unsqueeze(0).unsqueeze(0)  # 形状: (1, 1, H, W)
            mask = mask.expand(B, C, H, W)  # 形状: (B, C, H, W)
            mask = mask.to(image.device)  # 将掩码移动到与图像相同的设备

        # 应用掩码
        dft = dft * mask

        # 分离实部和虚部
        real_part = dft.real  # 实部，形状: (B, C, H, W)
        imag_part = dft.imag  # 虚部，形状: (B, C, H, W)

        # 拼接实部和虚部
        FFT_filtered = torch.cat((real_part, imag_part), dim=1)  # 形状: (B, 2 * C, H, W)
        return FFT_filtered