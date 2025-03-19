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
            # Training parameters
            time_steps=1000,  # Time step size, starting from 1 to 1000
            linear_start=1e-4,  # The starting point for linear beta
            linear_end=2e-2,  # The end point of linear beta
            base_learning_rate=1.0e-4,  # Initial learning rate
            batch_size=16,
            epoch=10,
            image_directory='data/train/flower/img/jpg',
            text_directory='data/train/flower/text_flower',
            img_h=512,  # Height of the image
            img_w=512,  # Width of the image
            out_path="output",  # Path to the output image
            out_num=2,  # Number of images to output
            vlb_Switch=True,  # Whether to enable the ELBO re-structuring loss
            device="cuda",  # Equipment
            pretrain_device="cuda",  # Location of the pre-trained model
            v_posterior=0,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
            original_elbo_weight=0,  # A hyperparameter that controls the weight of the ELBO loss term in the total loss
            snap=20,

            z_coef=0,  # Frequency domain information coefficient, range from 0 to 1,0 is pure sd and 1 is pure hyper
            z_in=6,
            z_dim=64,
            z_layer=6,

            mask_type=0,  # Decide on the mask type: 1 is high-pass filtering, keeping high frequencies, 0 is no filtering, and -1 is low-pass filtering, keeping low frequencies
            mask_radius=30,  # The filter mask radius, depending on the image size, is 30 for 256 and 60 for 512
            gaussian_mask=False,  # Whether to use Gaussian mask or not

            gen_txt_path="flower_gen.txt",  # The location of the txt file used for generation

            # DDIM
            DDIM_Switch=True,  # Whether to enable DDIM accelerated sampling
            DDIM_timesteps=100,  # The time step used for DDIM sampling
            DDIM_eta=0,  # Uncertainty hyperparameter

            # VAE
            VAE_Switch=True,  # Enable VAE or not
            VAE_config_path= "/root/pre_model/VAE/config.json",
            VAE_path="/root/pre_model/VAE/VAE.safetensor",  # Location of the VAE pretrained model

            # CLIP
            CLIP_Switch=True,
            CLIP_path='pre_model/CLIP',

            # Unet
            Unet_config=None,
            **kwargs
    ):
        super().__init__()
        self.time_steps = time_steps
        self.linear_start = linear_start
        self.linear_end = linear_end
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
        self.pretrain_device = pretrain_device

        self.z_coef = z_coef
        self.z_in = z_in
        self.z_dim = z_dim
        self.z_layer = z_layer

        self.gen_txt_path = gen_txt_path

        self.mask_type = mask_type
        self.mask_radius = mask_radius
        self.gaussian_mask = gaussian_mask

        self.DDIM_Switch = DDIM_Switch
        self.DDIM_timesteps = DDIM_timesteps
        self.DDIM_eta = DDIM_eta

        self.VAE_path = VAE_path
        self.VAE_config_path = VAE_config_path
        self.VAE_Switch = VAE_Switch

        self.CLIP_Switch = CLIP_Switch
        self.CLIP_path = CLIP_path

        # get sampling parameters
        def to_torch(np_in):  # Convert the np array to torch and set the precision and device
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
            step_size = self.time_steps // self.DDIM_timesteps
            DDIM_timesteps_in_DDPM = np.asarray(list(range(0, self.time_steps, step_size)))
            DDIM_timesteps_in_DDPM = DDIM_timesteps_in_DDPM + 1
            self.DDIM_timesteps_in_DDPM = to_torch(DDIM_timesteps_in_DDPM)  # The time step mapping of DDIM at DDPM

            DDIM_alphas = alphas[DDIM_timesteps_in_DDPM]
            DDIM_alphas_cumprod = alphas_cumprod[DDIM_timesteps_in_DDPM]
            DDIM_alphas_cumprod_prev = np.asarray(
                [alphas_cumprod[0]] + alphas_cumprod[DDIM_timesteps_in_DDPM[:-1]].tolist())
            DDIM_sigma = self.DDIM_eta * np.sqrt((1 - DDIM_alphas_cumprod_prev) / (1 - DDIM_alphas_cumprod) * (
                        1 - DDIM_alphas_cumprod / DDIM_alphas_cumprod_prev))
            self.DDIM_alphas = to_torch(DDIM_alphas)
            self.DDIM_alphas_cumprod = to_torch(DDIM_alphas_cumprod)
            self.DDIM_alphas_cumprod_prev = to_torch(DDIM_alphas_cumprod_prev)
            self.DDIM_sigma = to_torch(DDIM_sigma)

        # The adjusted shape is used for tensor broadcasting
        self.betas = self.betas.view(-1, 1, 1, 1)
        self.alphas = self.alphas.view(-1, 1, 1, 1)
        self.alphas_cumprod = self.alphas_cumprod.view(-1, 1, 1, 1)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.view(-1, 1, 1, 1)
        self.betas_ = self.betas.view(-1, 1, 1, 1)

        if 0 < z_coef <= 1:
            self.embed = ToVectorEmbedding(in_dim=z_in, out_dim=z_dim, layernum=z_layer).to(self.device)    # When the extra information coefficient is greater than 1, the coding layer is added

        if self.DDIM_Switch:
            self.DDIM_alphas = self.DDIM_alphas.view(-1, 1, 1, 1)
            self.DDIM_alphas_cumprod = self.DDIM_alphas_cumprod.view(-1, 1, 1, 1)
            self.DDIM_alphas_cumprod_prev = self.DDIM_alphas_cumprod_prev.view(-1, 1, 1, 1)
            self.DDIM_sigma = self.DDIM_sigma.view(-1, 1, 1, 1)

        # Acquiring VAE model
        if self.VAE_Switch:
            self.VAE = AutoencoderKL.from_single_file(self.VAE_path, config=self.VAE_config_path).to(self.pretrain_device).eval()
            for param in self.VAE.parameters():  # The parameters of the VAE are frozen
                param.requires_grad = False

        # Getting the CLIP model
        if self.CLIP_Switch:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                self.CLIP_path)
            self.transformer = CLIPTextModel.from_pretrained(self.CLIP_path).to(self.pretrain_device).eval()

        # Get the Unet model
        self.Unet = UNetModel(**Unet_config.get("params", dict())).to(self.device)

        # Obtain the tensor shape of the latent space
        tmp = torch.rand((1, 3, self.img_h, self.img_w), device=self.pretrain_device)
        self.latent_shape = self.encode(tmp).shape


    @torch.no_grad()
    # VAE encoding
    def encode(self, input):
        encoded_output = self.VAE.encode(input)
        latent_distribution = encoded_output.latent_dist
        latent = latent_distribution.sample()
        return latent

    @torch.no_grad()
    # VAE decoding
    def decode(self, latent):
        reconstructed_image = self.VAE.decode(latent).sample
        return reconstructed_image


    # Add noise in one step
    def q_sample(self, x_ori, t, noise):
        return torch.sqrt(self.alphas_cumprod[t]) * x_ori + torch.sqrt(1 - self.alphas_cumprod[t]) * noise

    # Single-step denoising
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
        nonzero_mask = (1 - (t == 0).float()).reshape(xt.shape[0], 1, 1, 1)  # When t=0, no noise is added, mask=0, else mask=1
        noise = torch.randn(xt.shape, device=self.device)
        xt_minus_1 = miu_ + torch.sqrt(beta_) * noise * nonzero_mask
        return xt_minus_1

    # Single-step denoising for DDIM
    @torch.no_grad()
    def DDIM_p_sample(self, xk, k, cond_txt=None):
        t = self.DDIM_timesteps_in_DDPM[k]
        if cond_txt is not None:
            noise_k = self.Unet(xk, t, cond_txt)
        else:
            noise_k = self.Unet(xk, t)
        nonzero_mask = (1 - (k == 0).float()).reshape(xk.shape[0], 1, 1, 1)  # When t=0, no noise is added, mask=0, else mask=1
        noise = torch.randn(xk.shape, device=self.device) * nonzero_mask
        xs = (torch.sqrt(self.DDIM_alphas_cumprod_prev[k]) * (
                    xk - torch.sqrt(1 - self.DDIM_alphas_cumprod[k]) * noise_k) / torch.sqrt(
            self.DDIM_alphas_cumprod[k]) +
              torch.sqrt(1 - self.DDIM_alphas_cumprod_prev[k] - self.DDIM_sigma[k] ** 2) * noise_k +
              self.DDIM_sigma[k] * noise)
        return xs

    # Denoising cycle
    @torch.no_grad()
    def p_sample_loop(self, img, cond_txt=None):
        img_cur = img  # Noisy image of the current number of steps
        if self.DDIM_Switch:
            for i in tqdm(reversed(range(0, self.DDIM_timesteps)), desc='Sampling t', unit='it',
                          total=self.DDIM_timesteps):
                batch_size = img.shape[0]
                img_cur = self.DDIM_p_sample(img_cur,
                                             torch.full((batch_size,), i, device=self.device, dtype=torch.long),
                                             cond_txt)
        else:
            for i in tqdm(reversed(range(0, self.time_steps)), desc='Sampling t', unit='it', total=self.time_steps):
                batch_size = img.shape[0]
                img_cur = self.p_sample(img_cur, torch.full((batch_size,), i, device=self.device, dtype=torch.long),
                                        cond_txt)
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
                transforms.Resize((self.img_h, self.img_w)),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1)
            ])
            optimizer = Adam(self.parameters(), lr=self.base_learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, cooldown=5, min_lr=0.0001)

            image_num_fid = 0
            image_knum_fid = 0
            image_num_clip = 0
            image_knum_clip = 0

            # Checkpoint path
            checkpoint_dir = "/root/autodl-tmp/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)  # Create the checkpoint directory
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_latest.pth")

            # If the checkpoint exists, load the model parameters and optimizer state
            if os.path.exists(checkpoint_path):
                print(f"Loading checkpoints: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path)
                if 0 < self.z_coef <= 1:
                    self.embed = self.embed.to("cpu")
                    self.embed(torch.randn(self.batch_size, 6, self.img_h, self.img_w))
                    self.embed = self.embed.to("cuda")
                self.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                if start_epoch % self.snap == 0:
                    start_epoch = start_epoch + 1
                best_loss = checkpoint['best_loss']
                print(f"from epoch {start_epoch} recovery training, loss: {best_loss}")
            else:
                print("No checkpoint found, train from scratch")
                start_epoch = 0
                best_loss = float('inf')  # Initialize the best loss

            for epoch in range(start_epoch, self.epoch):
                print(f"=============== epoch: {epoch} ===================")
                total_loss = 0.0

                if(epoch%self.snap==0 and epoch !=0):
                    gen_txts = random_select_lines(self.gen_txt_path, 10)  # Random n lines of description are selected
                    i = int(epoch)
                    for gen_txt in gen_txts:
                        self.predict(gen_txt, target_num=1, x=i)  # Generate n images
                        i = i + 1
                    # Saving the model
                    best_loss = avg_loss
                    torch.save({
                        'epoch': epoch-1,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_loss': best_loss,
                    }, checkpoint_path)
                    print(f"Model saved: {checkpoint_path}")

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
                        transform_pm = FFTtransform()  # FFT transform
                        if self.mask_type == 0:
                            batch_pm = transform_pm.ffttrans(batch_img)
                        elif self.mask_type == 1:
                            batch_pm = transform_pm.high_pass_filter(batch_img, self.mask_radius, self.gaussian_mask)
                        elif self.mask_type == -1:
                            batch_pm = transform_pm.low_pass_filter(batch_img, self.mask_radius, self.gaussian_mask)
                        batch_pm = self.embed(batch_pm)

                    if self.VAE_Switch:
                        Unet_in = self.encode(batch_img.half().to(self.device))  # VAE Encoding into latent space
                    else:
                        Unet_in = batch_img  # Without VAE encoding, the input image is the input of Unet
                    Unet_in = Unet_in.to(self.device)
                    t = torch.randint(0, self.time_steps, (Unet_in.shape[0],),
                                      device=self.device).long()  # Generate a batch of random time steps t, where t is a tensor
                    noise_target = torch.randn_like(Unet_in)  # A noise is randomly generated and used to add noise in the latent space
                    xt = self.q_sample(Unet_in, t, noise=noise_target)  # Image after adding noise
                    if self.CLIP_Switch:
                        tokens = self.tokenizer(batch_txt,
                                                max_length=77,
                                                padding='max_length',
                                                truncation=True,  # Explicitly enable truncation
                                                truncation_strategy='longest_first',  # Specify the truncation strategy
                                                return_tensors="pt"
                                                )
                        tokens.to(self.device)
                        c_embedding = self.transformer(**tokens).last_hidden_state
                        c_embedding = c_embedding.to(self.device)
                        noise_pred = self.Unet(xt, t, c_embedding, batch_pm)
                    else:
                        noise_pred = self.Unet(xt, t, None, batch_pm)

                    loss = self.get_loss(noise_target, noise_pred, t)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    total_loss += loss.item()
                avg_loss = total_loss / len(dataloader)
                scheduler.step(avg_loss)
                print("Loss:", avg_loss, "  learn_rate:", optimizer.param_groups[0]['lr'])
            # 保存模型
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
            }, checkpoint_path)
            print(f"Model saved: {checkpoint_path}")

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
                    device=self.device)  # The noise of a latent space is generated
            else:
                gaussian_noise = torch.randn((outnum, 3, self.img_h, self.img_w), device=self.device)

            c_embedding = None
            if self.CLIP_Switch:
                tokens = self.tokenizer(batch_txt,
                                        max_length=77,
                                        padding='max_length',
                                        truncation=True,
                                        truncation_strategy='longest_first',
                                        return_tensors="pt"
                                        )
                tokens.to(self.device)
                c_embedding = self.transformer(**tokens).last_hidden_state
                c_embedding = c_embedding.to(self.device)
                c_embedding_expanded = c_embedding.expand(outnum, -1, -1)
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

            # Check if the folder exists and create it if not
            if not os.path.exists(outpath):
                os.makedirs(outpath)
                print(f"Folder '{outpath}' created。")
            for i in range(outnum):
                img = sample[i]
                img = img.to("cpu")
                img = img.to(dtype=torch.float32)
                image_scale = (img + 1) / 2  # Rescaling [-1, 1] to [0, 1]
                img_np = (image_scale.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                image_generation = Image.fromarray(img_np)
                image_generation.save(outpath + '/' + f'output_image_{x}_{i}.png')
