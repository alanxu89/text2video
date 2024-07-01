class Config:

    def __init__(self) -> None:
        self.debug = False

        self.override_preprocessed = False
        self.preprocess_batch_size = 1
        self.preprocessed_dir = "encoder_out_16"
        self.text_key = "short_text"  # "text", "short_text", "category"

        self.use_preprocessed_data = True

        self.num_frames = 16
        self.frame_interval = 4
        self.image_size = (256, 256)

        self.noise_schedule = "linear"  #"squaredcos_cap_v2"

        self.use_videoldm = False
        self.image_finetune = False
        self.use_attention_mask = False

        # for video ldm temporal layers
        self.num_temp_heads = 8

        # self.depth = 24
        # self.hidden_size = 1024
        # self.num_heads = 16
        self.depth = 12
        self.hidden_size = 768
        self.num_heads = 12

        self.patch_size = (1, 2, 2)

        self.enable_temporal_attn = True
        self.joint_st_attn = False
        self.use_3dconv = False
        self.enable_mem_eff_attn = False
        self.enable_flashattn = True
        self.enable_grad_ckpt = True

        # for classifier-free guidance
        self.token_drop_prob = 0.1

        self.use_ema = True

        # pretrained vae
        self.vae_pretrained = "stabilityai/sd-vae-ft-ema"
        self.subfolder = ""
        # self.vae_pretrained = "madebyollin/sdxl-vae-fp16-fix"
        # self.subfolder = ""
        # self.vae_pretrained = "runwayml/stable-diffusion-v1-5"
        # self.subfolder = "vae"

        # text encoder
        self.textenc_pretrained = "DeepFloyd/t5-v1_1-xxl"
        self.model_max_length = 256  # only ~1% of captions exceeds this number
        self.text_encoder_output_dim = 4096
        # self.textenc_pretrained = "runwayml/stable-diffusion-v1-5"
        # self.model_max_length = 77
        # self.text_encoder_output_dim = 768

        # Define dataset
        # self.root = "/home/ubuntu/Documents/webvid/data/videos"
        # self.data_path = "/home/ubuntu/Downloads/data_train_partitions_0000_4/1.csv"
        self.root = "/mnt/sda/open_sora_plan_data"
        self.data_path = "mixkit.csv"

        self.use_image_transform = False
        self.num_workers = 4

        # Define acceleration
        self.dtype = "fp32"
        self.grad_checkpoint = True
        self.plugin = "zero2"
        self.sp_size = 1

        # Others
        self.seed = 123
        self.outputs = "outputs"
        self.wandb = False

        self.epochs = 3000
        self.log_every = 1
        self.ckpt_every = 10000
        self.load = None  #"outputs/027/checkpoints/0039999.pt"
        self.load_weights_only = False
        self.accum_iter = 1

        self.batch_size = 2
        self.lr = 3e-5
        self.grad_clip = 1.0

        # inference
        self.save_dir = "outputs/samples/"
        self.ckpt_path = "/home/ubuntu/Downloads/0099999.pt"  #"outputs/029/checkpoints/0004999.pt"
        self.cfg_scale = 4.0
        self.inference_sampling_steps = 250
