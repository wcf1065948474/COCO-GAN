import torch
import os
import torchvision.transforms as transforms
from PIL import Image



class MacroCelebaDataset(object):
    def __init__(self,opt):
        self.opt = opt
        self.wh = int(opt.full_size/opt.micro_size)-1
        self.celeba_imgpaths = self.get_img_path()
        self.size = len(self.celeba_imgpaths)
        self.transform = transforms.Compose(
            [transforms.CenterCrop(opt.full_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                (0.5, 0.5, 0.5))]
        )
    
    def get_img_path(self):
        imgpaths = []
        count = 0
        assert os.path.isdir(self.opt.datadir),'%s不是目录'%self.opt.datadir
        for root,_,fnames in os.walk(self.opt.datadir):
            for fname in fnames:
                path = os.path.join(root,fname)
                imgpaths.append(path)
                if self.opt.max_dataset != 0:
                    count += 1
                    if count == self.opt.max_dataset:
                        break
        return imgpaths    
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

class MicroCelebaDataset(object):
    def __init__(self,opt):
        self.opt = opt
        self.wh = 4 #
        self.celeba_imgpaths = self.get_img_path()
        self.size = len(self.celeba_imgpaths)
        self.transform = transforms.Compose(
            [transforms.CenterCrop(opt.full_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                (0.5, 0.5, 0.5))]
        )
    
    def get_img_path(self):
        imgpaths = []
        count = 0
        assert os.path.isdir(self.opt.datadir),'%s不是目录'%self.opt.datadir
        for root,_,fnames in os.walk(self.opt.datadir):
            for fname in fnames:
                path = os.path.join(root,fname)
                imgpaths.append(path)
                if self.opt.max_dataset != 0:
                    count += 1
                    if count == self.opt.max_dataset:
                        break
        return imgpaths    
    def __getitem__(self,index):
        self.micro_patches = []
        path = self.celeba_imgpaths[index % self.size]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        for i in range(self.wh):
            i *= self.opt.micro_size
            for j in range(self.wh):    
                j *= self.opt.micro_size
                patch = img[:,i:i+self.opt.micro_size,j:j+self.opt.micro_size]
                self.micro_patches.append(patch)
        return self.micro_patches
    def __len__(self):
        return self.size