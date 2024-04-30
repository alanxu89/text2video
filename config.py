class Config:

    def __init__(self) -> None:
        self.debug = False

        self.use_preprocessed_data = True
        self.override_preprocessed = False
        self.preprocess_batch_size = 8

        self.num_frames = 12
        self.frame_interval = 5
        self.image_size = (256, 256)

        self.depth = 12
        self.hidden_size = 768
        self.num_heads = 12
        self.patch_size = (1, 2, 2)

        self.joint_st_attn = True
        self.use_3dconv = False
        self.enable_mem_eff_attn = True
        self.enable_flashattn = False
        self.enable_grad_ckpt = True

        self.use_ema = False

        # pretrained vae
        # self.vae_pretrained = "stabilityai/sd-vae-ft-ema"
        # self.vae_scaling_factor = 0.18215
        self.vae_pretrained = "madebyollin/sdxl-vae-fp16-fix"
        self.vae_scaling_factor = 0.13025

        # text encoder
        self.textenc_pretrained = "DeepFloyd/t5-v1_1-xxl"
        self.model_max_length = 512

        # Define dataset
        # self.root = "/home/ubuntu/Documents/webvid/data/videos"
        # self.data_path = "/home/ubuntu/Downloads/data_train_partitions_0000_4/1.csv"
        self.root = "/mnt/sda/open_sora_plan_data"
        self.preprocessed_dir = "encoder_out"
        self.data_path = "mixkit.csv"

        self.use_image_transform = False
        self.num_workers = 8

        # Define acceleration
        self.dtype = "fp32"
        self.grad_checkpoint = True
        self.plugin = "zero2"
        self.sp_size = 1

        # Others
        self.seed = 123456
        self.outputs = "outputs"
        self.wandb = False

        self.epochs = 75
        self.log_every = 2
        self.ckpt_every = 10000
        self.load = None
        self.accum_iter = 4

        self.batch_size = 16
        self.lr = 3e-5
        self.grad_clip = 1.0

        self.save_dir = "outputs/samples/"
        self.ckpt_path = "outputs/003/checkpoints/0049999.pt"
