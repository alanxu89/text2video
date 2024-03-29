class Config:

    def __init__(self) -> None:
        self.num_frames = 16
        self.frame_interval = 3
        self.image_size = (256, 256)

        # Define dataset
        self.root = None
        # self.data_path = "/home/ubuntu/Downloads/data_train_partitions_0000.csv"
        self.data_path = "/home/ubuntu/Downloads/data_train_partitions_0000_4/1.csv"
        self.use_image_transform = False
        self.num_workers = 4

        # Define acceleration
        self.dtype = "bf16"
        self.grad_checkpoint = True
        self.plugin = "zero2"
        self.sp_size = 1

        # Others
        self.seed = 123456
        self.outputs = "outputs"
        self.wandb = False

        self.epochs = 1000
        self.log_every = 10
        self.ckpt_every = 1000
        self.load = None

        self.batch_size = 8
        self.lr = 2e-5
        self.grad_clip = 1.0
