from ..Unet import *
from ..score.random_select_lines import *
from diffusers import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm
from torchvision import transforms
from torch.amp import GradScaler, autocast
from torchvision.transforms import Compose
from torch.optim import Adam
from torch.utils.data import DataLoader


class FDDM(nn.Module):
    def __init__(
            self,
            # 训练参数
            time_steps=1000,  # 时间步长度，从1~1000开始
            linear_start=1e-4,  # 线性beta的起点
            linear_end=2e-2,  # 线性beta的终点
            base_learning_rate=1.0e-4,  # 初始学习率
            batch_size=16,
            epoch=10,
            image_directory='data/train/flower/img/jpg',
            text_directory='data/train/flower/text_flower',
            img_h=512,  # 图像的高
            img_w=512,  # 图像的宽
            out_path="output",  # 输出图像的路径
            out_num=2,  # 输出图像的个数
            vlb_Switch=True,  # 是否启用ELBO重新构损失
            device="cuda",  # 设备
            pretrain_device="cuda",  # 预训练模型的位置
            v_posterior=0,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
            original_elbo_weight=0,  # 一个超参数，用于控制 ELBO 损失项在总损失中的权重
            snap=20,

            z_coef=0,  # 频域信息系数，范围1-0，0为纯sd，1为纯hyper
            z_in=6,
            z_dim=64,
            z_layer=6,

            mask_type=0,  # 决定掩码类型，1为高通滤波，保留高频，0为不做滤波处理，-1为低通滤波，保留低频
            mask_radius=30,  # 滤波掩码半径，取决于图像尺寸，256取30，512取60
            gaussian_mask=False,    # 是否采用高斯掩码

            gen_txt_path="flower_gen.txt",      # 生成所用的txt文件的位置

            # DDIM
            DDIM_Switch=True,  # 是否启用DDIM加速采样
            DDIM_timesteps=100,  # DDIM采样所用的时间步
            DDIM_eta=0,  # 不确定性超参

            # VAE
            VAE_Switch=True,  # 是否启用VAE
            VAE_config_path= "/root/pre_model/VAE/config.json",
            VAE_path="/root/pre_model/VAE/VAE.safetensor",  # VAE预训练模型的位置

            # CLIP
            CLIP_Switch=True,
            CLIP_path='pre_model/CLIP',

            # Unet
            Unet_config=None,
            **kwargs
    ):
        super().__init__()
        self.time_steps = time_steps  # 时间步长度，从1~1000开始
        self.linear_start = linear_start  # 线性beta的起点
        self.linear_end = linear_end  # 线性beta的终点
        self.base_learning_rate = base_learning_rate
        self.batch_size = batch_size
        self.epoch = epoch
        self.image_directory = image_directory
        self.text_directory = text_directory
        self.img_h = img_h
        self.img_w = img_w
        self.vlb_Switch = vlb_Switch
        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight

        self.snap = snap
        self.out_path = out_path
        self.out_num = out_num

        self.device = device
        self.pretrain_device = pretrain_device  # 预训练模型的位置

        self.z_coef = z_coef    # 频域信息系数，范围1-0，0为纯sd，1为纯hyper
        self.z_in = z_in
        self.z_dim = z_dim
        self.z_layer = z_layer

        self.gen_txt_path = gen_txt_path  # 生成所用的txt文件的位置

        self.mask_type = mask_type  # 决定掩码类型，1为高通滤波，保留高频，0为不做滤波处理，-1为低通滤波，保留低频
        self.mask_radius = mask_radius  # 滤波掩码半径，取决于图像尺寸，256取30，512取60
        self.gaussian_mask = gaussian_mask

        self.DDIM_Switch = DDIM_Switch
        self.DDIM_timesteps = DDIM_timesteps
        self.DDIM_eta = DDIM_eta

        self.VAE_path = VAE_path
        self.VAE_config_path = VAE_config_path
        self.VAE_Switch = VAE_Switch

        self.CLIP_Switch = CLIP_Switch
        self.CLIP_path = CLIP_path

        # 获取采样参数
        def to_torch(np_in):  # 将np数组转化为torch，并设置精度与设备
            return torch.tensor(np_in, dtype=torch.float32, device=self.device)

        betas = np.linspace(self.linear_start ** 0.5, self.linear_end ** 0.5, self.time_steps,
                            dtype=np.float32) ** 2
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        betas_ = (1 - self.v_posterior) * betas * (1 - alphas_cumprod_prev) / (
                    1 - alphas_cumprod) + self.v_posterior * betas
        self.betas = to_torch(betas)
        self.alphas = to_torch(alphas)
        self.alphas_cumprod = to_torch(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch(alphas_cumprod_prev)
        self.betas_ = to_torch(betas_)

        lvlb_weights = self.betas ** 2 / (2 * self.betas_ * self.alphas * (1 - self.alphas_cumprod))
        lvlb_weights[0] = lvlb_weights[1]
        self.lvlb_weights = lvlb_weights

        if self.DDIM_Switch:
            step_size = self.time_steps // self.DDIM_timesteps  # 计算每个 DDIM 时间步对应多少个 DDPM 时间步。c 是一个整数，表示每 c 个 DDPM 时间步中选择一个作为 DDIM 时间步。
            DDIM_timesteps_in_DDPM = np.asarray(list(range(0, self.time_steps, step_size)))
            DDIM_timesteps_in_DDPM = DDIM_timesteps_in_DDPM + 1
            self.DDIM_timesteps_in_DDPM = to_torch(DDIM_timesteps_in_DDPM)  # DDIM在DDPM的时间步映射

            DDIM_alphas = alphas[DDIM_timesteps_in_DDPM]
            DDIM_alphas_cumprod = alphas_cumprod[DDIM_timesteps_in_DDPM]
            # DDIM_alphas_cumprod_prev = alphas_cumprod_prev[DDIM_timesteps_in_DDPM]
            DDIM_alphas_cumprod_prev = np.asarray(
                [alphas_cumprod[0]] + alphas_cumprod[DDIM_timesteps_in_DDPM[:-1]].tolist())
            # DDIM_sigma = self.DDIM_eta * np.sqrt((1 - DDIM_alphas_cumprod_prev) * (1 - DDIM_alphas) / (1 - DDIM_alphas_cumprod))
            DDIM_sigma = self.DDIM_eta * np.sqrt((1 - DDIM_alphas_cumprod_prev) / (1 - DDIM_alphas_cumprod) * (
                        1 - DDIM_alphas_cumprod / DDIM_alphas_cumprod_prev))
            self.DDIM_alphas = to_torch(DDIM_alphas)
            self.DDIM_alphas_cumprod = to_torch(DDIM_alphas_cumprod)
            self.DDIM_alphas_cumprod_prev = to_torch(DDIM_alphas_cumprod_prev)
            self.DDIM_sigma = to_torch(DDIM_sigma)

        # 调整形状用于张量广播
        self.betas = self.betas.view(-1, 1, 1, 1)
        self.alphas = self.alphas.view(-1, 1, 1, 1)
        self.alphas_cumprod = self.alphas_cumprod.view(-1, 1, 1, 1)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.view(-1, 1, 1, 1)
        self.betas_ = self.betas.view(-1, 1, 1, 1)

        if 0 < z_coef <= 1:
            self.embed = ToVectorEmbedding(in_dim=z_in, out_dim=z_dim, layernum=z_layer).to(self.device)    # 额外信息系数大于1则添加编码层

        if self.DDIM_Switch:
            self.DDIM_alphas = self.DDIM_alphas.view(-1, 1, 1, 1)
            self.DDIM_alphas_cumprod = self.DDIM_alphas_cumprod.view(-1, 1, 1, 1)
            self.DDIM_alphas_cumprod_prev = self.DDIM_alphas_cumprod_prev.view(-1, 1, 1, 1)
            self.DDIM_sigma = self.DDIM_sigma.view(-1, 1, 1, 1)

        # 获取VAE模型
        if self.VAE_Switch:
            self.VAE = AutoencoderKL.from_single_file(self.VAE_path, config=self.VAE_config_path).to(self.pretrain_device).eval()
            for param in self.VAE.parameters():  # 冻结VAE的参数
                param.requires_grad = False

        # 获取CLIP模型
        if self.CLIP_Switch:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                self.CLIP_path)  # 分词器, 将原始文本（如句子或段落）分割成一系列的标记（tokens）, 每个标记对应一个唯一的 ID(文本编码# )
            self.transformer = CLIPTextModel.from_pretrained(self.CLIP_path).to(self.pretrain_device).eval()  # 将文本编码为高维向量表示。它接收由 CLIPTokenizer 生成的标记化输入

        # 获取Unet模型
        # model = UNetModel(**config.get("model").get("params", dict()))
        self.Unet = UNetModel(**Unet_config.get("params", dict())).to(self.device)

        # 获取潜在空间的张量形状
        tmp = torch.rand((1, 3, self.img_h, self.img_w), device=self.pretrain_device)
        self.latent_shape = self.encode(tmp).shape


    @torch.no_grad()
    # VAE编码
    def encode(self, input):
        encoded_output = self.VAE.encode(input)  # 返回 AutoencoderKLOutput 对象
        latent_distribution = encoded_output.latent_dist  # 获取 DiagonalGaussianDistribution
        latent = latent_distribution.sample()  # 从分布中采样得到潜变量
        return latent

    @torch.no_grad()
    # VAE解码
    def decode(self, latent):
        reconstructed_image = self.VAE.decode(latent).sample
        return reconstructed_image


    # 一步加噪
    def q_sample(self, x_ori, t, noise):
        return torch.sqrt(self.alphas_cumprod[t]) * x_ori + torch.sqrt(1 - self.alphas_cumprod[t]) * noise

    # 单步去噪
    @torch.no_grad()
    def p_sample(self, xt, t, cond_txt=None):
        if cond_txt is not None:
            noise_t = self.Unet(xt, t, cond_txt)
        else:
            noise_t = self.Unet(xt, t)
        miu_ = (xt - (1 - self.alphas[t]) * noise_t / torch.sqrt(1 - self.alphas_cumprod[t])) / torch.sqrt(
            self.alphas[t])
        v_posterior = self.v_posterior
        beta_ = (1 - v_posterior) * self.betas[t] * (1 - self.alphas_cumprod_prev[t]) / (
                    1 - self.alphas_cumprod[t]) + v_posterior * self.betas[t]
        nonzero_mask = (1 - (t == 0).float()).reshape(xt.shape[0], 1, 1, 1)  # 当t=0时，不添加噪声，mask=0，else mask=1
        noise = torch.randn(xt.shape, device=self.device)
        xt_minus_1 = miu_ + torch.sqrt(beta_) * noise * nonzero_mask
        return xt_minus_1

    # DDIM的单步去噪
    @torch.no_grad()
    def DDIM_p_sample(self, xk, k, cond_txt=None):
        t = self.DDIM_timesteps_in_DDPM[k]
        if cond_txt is not None:
            noise_k = self.Unet(xk, t, cond_txt)
        else:
            noise_k = self.Unet(xk, t)
        nonzero_mask = (1 - (k == 0).float()).reshape(xk.shape[0], 1, 1, 1)  # 当t=0时，不添加噪声，mask=0，else mask=1
        noise = torch.randn(xk.shape, device=self.device) * nonzero_mask
        xs = (torch.sqrt(self.DDIM_alphas_cumprod_prev[k]) * (
                    xk - torch.sqrt(1 - self.DDIM_alphas_cumprod[k]) * noise_k) / torch.sqrt(
            self.DDIM_alphas_cumprod[k]) +
              torch.sqrt(1 - self.DDIM_alphas_cumprod_prev[k] - self.DDIM_sigma[k] ** 2) * noise_k +
              self.DDIM_sigma[k] * noise)
        return xs

    # 去噪循环
    @torch.no_grad()
    def p_sample_loop(self, img, cond_txt=None):
        img_cur = img  # 当前步数的噪声图像
        if self.DDIM_Switch:
            for i in tqdm(reversed(range(0, self.DDIM_timesteps)), desc='Sampling t', unit='it',
                          total=self.DDIM_timesteps):
                batch_size = img.shape[0]
                img_cur = self.DDIM_p_sample(img_cur,
                                             torch.full((batch_size,), i, device=self.device, dtype=torch.long),
                                             cond_txt)  # 要将时间步与批量匹配
        else:
            for i in tqdm(reversed(range(0, self.time_steps)), desc='Sampling t', unit='it', total=self.time_steps):
                batch_size = img.shape[0]
                img_cur = self.p_sample(img_cur, torch.full((batch_size,), i, device=self.device, dtype=torch.long),
                                        cond_txt)  # 要将时间步与批量匹配
        return img_cur

    def get_loss(self, noise_target, noise_pred, t):
        if self.vlb_Switch:
            simple_loss = torch.nn.functional.mse_loss(noise_target, noise_pred)
            loss_vlb = (self.lvlb_weights[t] * simple_loss).mean()
            loss = simple_loss + self.original_elbo_weight * loss_vlb
        else:
            simple_loss = torch.nn.functional.mse_loss(noise_target, noise_pred).mean()
            loss = simple_loss
        return loss

    def train_step(self):
        self.scaler = GradScaler()
        with autocast(device_type=self.device):
            transform = Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((self.img_h, self.img_w)),  # 调整图像大小
                # transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),  # 将图像转换为张量
                transforms.Lambda(lambda t: (t * 2) - 1)  # 将张量的值从 [0, 1] 映射到 [-1, 1]
            ])
            # dict_text = get_dict_text(self.text_directory)
            # dataset = txt2img_Dataset(self.image_directory, dict_text, transform=transform)
            # dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            # print("data have loaded")
            optimizer = Adam(self.parameters(), lr=self.base_learning_rate)
            # scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, cooldown=5, min_lr=0.0001)

            image_num_fid = 0
            image_knum_fid = 0
            image_num_clip = 0
            image_knum_clip = 0

            # 检查点路径
            checkpoint_dir = "/root/autodl-tmp/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)  # 创建检查点目录
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_latest.pth")

            # 如果检查点存在，加载模型参数和优化器状态
            if os.path.exists(checkpoint_path):
                print(f"加载检查点: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path)
                if 0 < self.z_coef <= 1:
                    self.embed = self.embed.to("cpu")
                    self.embed(torch.randn(self.batch_size, 6, self.img_h, self.img_w))
                    self.embed = self.embed.to("cuda")
                self.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1  # 从下一个 epoch 开始
                if start_epoch % self.snap == 0:
                    start_epoch = start_epoch + 1
                best_loss = checkpoint['best_loss']
                print(f"从 epoch {start_epoch} 恢复训练，损失: {best_loss}")
            else:
                print("未找到检查点，从头开始训练")
                start_epoch = 0
                best_loss = float('inf')  # 初始化最佳损失

            for epoch in range(start_epoch, self.epoch):
                print(f"=============== epoch: {epoch} ===================")
                total_loss = 0.0

                if(epoch%self.snap==0 and epoch !=0):
                    gen_txts = random_select_lines(self.gen_txt_path, 10)  # 选取随机n行描述
                    i = int(epoch)
                    for gen_txt in gen_txts:
                        self.predict(gen_txt, target_num=1, x=i)  # 生成n张图片到fakepath
                        i = i + 1
                    # 保存模型
                    best_loss = avg_loss
                    torch.save({
                        'epoch': epoch-1,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_loss': best_loss,
                    }, checkpoint_path)
                    print(f"模型已保存: {checkpoint_path}")

                dict_text = get_dict_text(self.text_directory)
                dataset = txt2img_Dataset(self.image_directory, dict_text, transform=transform)
                dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
                print("data have loaded")
                for step, (batch_img, batch_txt) in tqdm(enumerate(dataloader), desc='batch', unit='it',
                                                         total=len(dataloader)):
                    optimizer.zero_grad()

                    if self.FID_score:
                        image_num_fid = image_num_fid+self.batch_size
                        if image_num_fid >= (self.FID_step * 1000):
                            image_num_fid = image_num_fid - self.FID_step * 1000
                            image_knum_fid = image_knum_fid + self.FID_step
                            with open("runlog.txt", "a") as file:
                                file.write(f"image_num: {image_knum_fid}k\n")
                            with open("runlog.txt", "a") as file:
                                file.write(f"Loss: {loss.item()}\n")
                            self.getFID()

                    if self.CLIP_score:
                        image_num_clip = image_num_clip+self.batch_size
                        if image_num_clip >= (self.CLIP_step * 1000):
                            image_num_clip = image_num_clip - self.CLIP_step * 1000
                            image_knum_clip = image_knum_clip + self.CLIP_step
                            with open("runlog.txt", "a") as file:
                                file.write(f"image_num: {image_knum_fid}k\n")
                            with open("runlog.txt", "a") as file:
                                file.write(f"Loss: {loss.item()}\n")
                            self.getCLIP()


                    batch_img = batch_img.to(self.device)

                    batch_pm = None
                    if 0 < self.z_coef <= 1:
                        transform_pm = FFTtransform()  # FFT 变换
                        if self.mask_type == 0:
                            batch_pm = transform_pm.ffttrans(batch_img)  # 输出形状: (B, 2 * C, H, W)  # 频域信息批次
                        elif self.mask_type == 1:
                            batch_pm = transform_pm.high_pass_filter(batch_img, self.mask_radius, self.gaussian_mask)
                        elif self.mask_type == -1:
                            batch_pm = transform_pm.low_pass_filter(batch_img, self.mask_radius, self.gaussian_mask)
                        batch_pm = self.embed(batch_pm)

                    if self.VAE_Switch:
                        Unet_in = self.encode(batch_img.half().to(self.device))  # VAE编码至潜在空间
                    else:
                        Unet_in = batch_img  # 不进行VAE编码，输入图像即为Unet的输入
                    Unet_in = Unet_in.to(self.device)
                    t = torch.randint(0, self.time_steps, (Unet_in.shape[0],),
                                      device=self.device).long()  # 生成一批随机的时间步t, t为一个张量
                    noise_target = torch.randn_like(Unet_in)  # 随机生成一个噪声，用于在潜在空间进行加噪
                    xt = self.q_sample(Unet_in, t, noise=noise_target)  # 得加噪后的图像
                    if self.CLIP_Switch:
                        tokens = self.tokenizer(batch_txt,
                                                max_length=77,
                                                padding='max_length',
                                                truncation=True,  # 显式启用截断
                                                truncation_strategy='longest_first',  # 指定截断策略
                                                return_tensors="pt"
                                                )
                        tokens.to(self.device)
                        c_embedding = self.transformer(**tokens).last_hidden_state
                        c_embedding = c_embedding.to(self.device)
                        noise_pred = self.Unet(xt, t, c_embedding, batch_pm)
                    else:
                        noise_pred = self.Unet(xt, t, None, batch_pm)

                    loss = self.get_loss(noise_target, noise_pred, t)
                    # print(loss.item())
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    total_loss += loss.item()
                    # loss.backward()
                    # optimizer.step()
                avg_loss = total_loss / len(dataloader)
                scheduler.step(avg_loss)
                print("Loss:", avg_loss, "  learn_rate:", optimizer.param_groups[0]['lr'])
                # print(f"Learning rate after epoch {epoch}: {scheduler.get_last_lr()}")
            # 保存模型
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
            }, checkpoint_path)
            print(f"模型已保存: {checkpoint_path}")

    @torch.no_grad()
    def predict(self, batch_txt, target_dict=None, target_num=None, x=0):
        with autocast(device_type=self.device):

            if target_num is None:
                outnum = self.out_num
            else:
                outnum = target_num

            if self.VAE_Switch:
                gaussian_noise = torch.randn(
                    (outnum, self.latent_shape[1], self.latent_shape[2], self.latent_shape[3]),
                    device=self.device)  # 生成一个潜在空间的噪声
            else:
                gaussian_noise = torch.randn((outnum, 3, self.img_h, self.img_w), device=self.device)

            c_embedding = None
            if self.CLIP_Switch:
                tokens = self.tokenizer(batch_txt,
                                        max_length=77,
                                        padding='max_length',
                                        truncation=True,  # 显式启用截断
                                        truncation_strategy='longest_first',  # 指定截断策略
                                        return_tensors="pt"
                                        )
                tokens.to(self.device)
                c_embedding = self.transformer(**tokens).last_hidden_state
                c_embedding = c_embedding.to(self.device)
                c_embedding_expanded = c_embedding.expand(outnum, -1, -1)  # 沿第一个维度扩展数据
            if self.VAE_Switch:
                latent_sample = self.p_sample_loop(gaussian_noise, c_embedding_expanded)
                latent_sample = latent_sample.to(self.pretrain_device)
                sample = self.decode(latent_sample)
            else:
                sample = self.p_sample_loop(gaussian_noise, c_embedding_expanded)

            if target_dict is None:
                outpath = self.out_path
            else:
                outpath = target_dict

            # 检查文件夹是否存在，如果不存在则创建
            if not os.path.exists(outpath):
                os.makedirs(outpath)
                print(f"文件夹 '{outpath}' 已创建。")
            # else:
                # print(f"文件夹 '{outpath}' 已存在。")
            for i in range(outnum):
                img = sample[i]
                img = img.to("cpu")
                img = img.to(dtype=torch.float32)
                image_scale = (img + 1) / 2  # 将 [-1, 1] 缩放到 [0, 1]
                img_np = (image_scale.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                image_generation = Image.fromarray(img_np)
                image_generation.save(outpath + '/' + f'output_image_{x}_{i}.png')
