import torch
import torch.nn as nn
import numpy as np
import option
import time
import matplotlib.pyplot as plt

class Timer():
  def __init__(self):
    self.seconds = 32400
    self.total_start_time = int(time.time())
  def get_start_time(self):
    self.start_time = int(time.time())
  def get_end_time(self):
    self.end_time = int(time.time())
  def spend_time(self):
    return self.end_time-self.start_time

class ConditionalBatchNorm2d(nn.Module):
  def __init__(self,opt,num_features):
    super().__init__()
    self.opt = opt
    self.num_features = num_features
    inter_dim = 2*num_features
    self.bn = nn.BatchNorm2d(num_features,affine=False)
    self.gamma_mlp = nn.Sequential(
      nn.utils.spectral_norm(nn.Linear(128,inter_dim)),
      nn.ReLU(),
      nn.utils.spectral_norm(nn.Linear(inter_dim,num_features)),
      nn.ReLU()
    )
    self.beta_mlp = nn.Sequential(
      nn.utils.spectral_norm(nn.Linear(128,inter_dim)),
      nn.ReLU(),
      nn.utils.spectral_norm(nn.Linear(inter_dim,num_features)),
      nn.ReLU()
    )
  def forward(self,x,y):
    out = self.bn(x)
    gamma = self.gamma_mlp(y)
    beta = self.beta_mlp(y)
    out = gamma.view(self.opt.batchsize*self.opt.micro_in_macro,self.num_features,1,1)*out + beta.view(self.opt.batchsize*self.opt.micro_in_macro,self.num_features,1,1)
    return out


# class ConditionalBatchNorm2d_OLD(nn.Module):
#   def __init__(self,opt, num_features):
#     super().__init__()
#     self.opt = opt
#     self.num_features = num_features
#     self.bn = nn.BatchNorm2d(num_features, affine=False)
#     self.embed = nn.Embedding(opt.num_classes, num_features * 2)
#     self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
#     self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

#   def forward(self, x, y):
#     out = self.bn(x)
#     ebd_y = self.embed(y).reapt((self.opt.batchsize,1))
#     gamma, beta = ebd_y.chunk(2, 1)
#     out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
#     return out

class GeneratorResidualBlock(nn.Module):
  def __init__(self,opt,input_channel,output_channel):
    super().__init__()
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    # self.upscale = nn.Upsample(scale_factor=2)
    # self.upscale_branch = nn.Upsample(scale_factor=2)
    self.conv1 = nn.utils.spectral_norm(nn.Conv2d(input_channel,output_channel,3,padding=1))
    self.conv2 = nn.utils.spectral_norm(nn.Conv2d(output_channel,output_channel,3,padding=1))
    self.conv_branch = nn.utils.spectral_norm(nn.Conv2d(input_channel,output_channel,3,padding=1))
    self.cbn = ConditionalBatchNorm2d(opt,output_channel)

  def forward(self,input,y):
    master = self.relu1(input)
    master = nn.functional.interpolate(master,scale_factor=2)
    master = self.conv1(master)
    master = self.cbn(master,y)
    master = self.relu2(master)
    master = self.conv2(master)
    branch = nn.functional.interpolate(input,scale_factor=2)
    branch = self.conv_branch(branch)
    return master+branch


class Generator(nn.Module):
  def __init__(self,opt):
    super().__init__()
    self.opt = opt
    self.linear = nn.utils.spectral_norm(nn.Linear(opt.latentsize+opt.y_ebdsize,opt.latentoutsize))
    self.grb1 = GeneratorResidualBlock(opt,1024,512)
    self.grb2 = GeneratorResidualBlock(opt,512,256)
    self.grb3 = GeneratorResidualBlock(opt,256,128)
    self.grb4 = GeneratorResidualBlock(opt,128,64)
    self.model = nn.Sequential(
      nn.utils.spectral_norm(nn.BatchNorm2d(64)),
      nn.ReLU(),
      nn.utils.spectral_norm(nn.Conv2d(64,3,3,padding=1)),
      nn.Tanh()
    )
  def forward(self,input,y):
    res = self.linear(input)
    res = res.view(self.opt.batchsize*self.opt.micro_in_macro,1024,2,2)
    res = self.grb1(res,y)
    res = self.grb2(res,y)
    res = self.grb3(res,y)
    res = self.grb4(res,y)
    res = self.model(res)
    return res

class DiscriminatorResidualBlock(nn.Module):
  def __init__(self,input_channel,output_channel,pooling=True):
    super().__init__()
    self.pooling = pooling
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.conv1 = nn.utils.spectral_norm(nn.Conv2d(input_channel,output_channel,3,padding=1))
    self.conv2 = nn.utils.spectral_norm(nn.Conv2d(output_channel,output_channel,3,padding=1))
    self.conv_branch = nn.utils.spectral_norm(nn.Conv2d(input_channel,output_channel,3,padding=1))
    if self.pooling == True:
      self.avg_pool = nn.AvgPool2d(2,2)
      self.avg_pool_branch = nn.AvgPool2d(2,2)

  def forward(self,input):
    master = self.relu1(input)
    master = self.conv1(master)
    master = self.relu2(master)
    master = self.conv2(master)
    if self.pooling == True:
      master = self.avg_pool(master)
      branch = self.avg_pool_branch(input)
      branch = self.conv_branch(branch)
    else:
      branch = self.conv_branch(input)
    return branch+master
    
class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.drb1 = DiscriminatorResidualBlock(3,64)
    self.drb2 = DiscriminatorResidualBlock(64,128)
    self.drb3 = DiscriminatorResidualBlock(128,256)
    self.drb4 = DiscriminatorResidualBlock(256,512)
    self.drb5 = DiscriminatorResidualBlock(512,512,False)
    self.relu = nn.ReLU()
    self.glb_pool = nn.AdaptiveMaxPool2d(1)
    self.linear = nn.utils.spectral_norm(nn.Linear(512,1))
    self.linear_branch = nn.utils.spectral_norm(nn.Linear(2,512))
    self.dah = nn.Sequential(
      nn.utils.spectral_norm(nn.BatchNorm1d(512)),
      nn.utils.spectral_norm(nn.Linear(512,128)),
      nn.BatchNorm1d(128),
      nn.LeakyReLU(),
      nn.utils.spectral_norm(nn.Linear(128,2)),#1->28
      nn.Tanh()
    )

  def forward(self,input,y):
    master = self.drb1(input)
    master = self.drb2(master)
    master = self.drb3(master)
    master = self.drb4(master)
    master = self.drb5(master)
    master = self.relu(master)
    master = self.glb_pool(master)
    master = torch.squeeze(master)
    h = self.dah(master)
    projection = self.linear_branch(y)
    projection = projection*master
    projection = torch.sum(projection,1,True)/512.0
    master = self.linear(master)
    return master+projection,h


# class GeneratePosList(object):
#   def __init__(self,opt):
#     self.opt=opt
#     self.wh = int(np.sqrt(opt.num_classes))
#     self.pos_table = torch.arange(opt.num_classes).view(self.wh,self.wh)
#     self.max_area = int(np.sqrt(opt.num_classes)-np.sqrt(opt.micro_in_macro)+1)
#     self.macro_table = self.pos_table[0:self.max_area,0:self.max_area]
#     self.macro_table = self.macro_table.contiguous().view(-1)
#   def get_pos_list(self,p,isMacro=True):
#     pos = []
#     if isMacro:
#       pos.append(self.macro_table[p])
#     else:
#       for i in range(int(np.sqrt(self.opt.micro_in_macro))):
#         for j in range(int(np.sqrt(self.opt.micro_in_macro))):
#           pos.append(self.macro_table[p]+i*self.wh+j)
#     return torch.LongTensor(pos)





# if __name__ == '__main__':
#   opt = option.Option()
#   gety = Get_Latent_NewY(opt)
#   gety.get_latent()
#   gety.get_micro_list(2)
#   tmp=gety.get_latent_ebdy(2)
#   print(tmp.size())
#   print(tmp[:5,126:])



# if __name__ == '__main__':
#   opt = option.Option()
#   opt.batchsize=16
#   G = Generator(opt)
#   D = Discriminator()
#   ebd = nn.Embedding(16,28)
#   y = torch.arange(16)
#   ebd_y = ebd(y)

#   z = torch.randn(1,100)
#   z = z.expand(16,-1)
#   z = torch.cat((z,ebd_y),1)
#   print(z.size())
#   G.eval()
#   D.eval()
#   a = G(z,y)

#   b,c = D(a,ebd_y)
#   a = a.detach().numpy()
#   a = a[0]
#   a = np.transpose(a,(1,2,0))
#   print(b,c)
#   plt.figure(figsize=(10,10))
#   plt.axis('off')
#   plt.imshow(a)
#   plt.show()
