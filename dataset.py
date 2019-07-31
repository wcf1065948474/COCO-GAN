import torch
import os
import torchvision.transforms as transforms
from PIL import Image

def get_img_path(dir):
    imgpaths = []
    assert os.path.isdir(dir),'%s不是目录'%dir
    for root,_,fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root,fname)
            imgpaths.append(path)
    return imgpaths

class MacroCelebaDataset(object):
    def __init__(self,opt):
        self.opt = opt
        self.wh = int(opt.full_szie/opt.micro_size)-1
        self.celeba_imgpaths = get_img_path(self.opt.datadir)
        self.size = len(self.celeba_imgpaths)
        self.transform = transforms.Compose(
            [transforms.CenterCrop(opt.full_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                (0.5, 0.5, 0.5))]
        )
        
    def __getitem__(self,index):
        self.macro_patches = []
        path = self.celeba_imgpaths[index % self.size]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        for i in range(self.wh):
            i *= self.opt.micro_size
            for j in range(self.wh):    
                j *= self.opt.micro_size
                patch = img[:,i:i+self.opt.macro_size,j:j+self.opt.macro_size]
                self.macro_patches.append(patch)
        return self.macro_patches
    def __len__(self):
        return self.size