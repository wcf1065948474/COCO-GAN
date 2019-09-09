class Option(object):
    def __init__(self):
        self.batchsize = 256
        self.latentsize = 100
        self.y_ebdsize = 28
        self.latentoutsize = 1024*2*2
        self.num_classes = 16
        self.micro_in_macro = 4
        self.macro_in_full = 4
        self.datadir='../input/img_align_celeba/img_align_celeba'
        self.macro_size = 64
        self.micro_size = 32
        self.full_size = 128
        self.LAMBDA = 10
        self.ALPHA = 100
        self.epoch = 50
        self.max_dataset = 0
        self.my_model_dir = 'my_model'