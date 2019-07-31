class Option(object):
    def __init__(self):
        self.batchsize = 64
        self.latentsize = 100
        self.y_emd_size = 28
        self.latentoutsize = 1024*2*2
        self.num_classes = 16
        self.micro_in_macro = 4
        self.macro_in_full = 4
        self.datadir='../input/img_align_celeba/img_align_celeba'
        self.macro_size = 64
        self.micro_size = 32
        self.full_szie = 128
        self.LAMBDA = 10
        self.ALPHA = 100
        self.epoch = 50