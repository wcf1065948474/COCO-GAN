import torch
import option
import models
import numpy as np

class GetLatentY(object):
    def __init__(self,opt):
        self.opt = opt
        self.ebd = torch.nn.Embedding(16,28)
    def get_new_latent(self):
        self.z = np.random.normal(0.0,1.0,(self.opt.batchsize,100)).astype(np.float32)
        self.z = np.repeat(self.z,self.opt.micro_in_macro,0)
        self.z = torch.from_numpy(self.z)
    def get_ebd(self,y):
        ebd_y = self.ebd(y)
        ebd_y = ebd_y.expand(self.opt.batchsize,-1)
        return ebd_y
    def get_latent_y(self,y):
        ebd_y = self.ebd(y)
        ebd_y = ebd_y.repeat((self.opt.batchsize,1))
        latent_y = torch.cat((self.z,ebd_y),1)
        return latent_y,ebd_y


class COCOGAN(object):
    def __init__(self,opt):
        self.opt = opt
        self.G = models.Generator(opt)
        self.D = models.Discriminator()
        self.Lsloss = torch.nn.MSELoss()
        self.optimizerG = torch.optim.SparseAdam(self.G.parameters(),1e-4,(0,0.999))
        self.optimizerD = torch.optim.SparseAdam(self.D.parameters(),4e-4,(0,0.999))
        self.d_losses = []
        self.g_losses = []
        self.G.cuda()
        self.D.cuda()
    def macro_from_micro(self,micro):
        hw = int(np.sqrt(self.opt.micro_in_macro))
        batchs = int(micro.size(0)/self.opt.micro_in_macro)
        spatial = int(self.opt.micro_size*hw)
        macros = torch.empty((batchs,3,spatial,spatial),dtype=micro.dtype)
        for b in range(batchs):
            bb = b*self.opt.micro_in_macro
            for i in range(hw):
                ii = i*self.opt.micro_size
                for j in range(hw):
                    jj = j*self.opt.micro_size
                    macros[b,:,ii:ii+self.opt.micro_size,jj:jj+self.opt.micro_size] = micro[bb+hw*i+j]
        return macros
    def calc_gradient_penalty(self,real_data,fake_data):
        #print real_data.size()
        alpha = torch.rand(self.opt.batchsize, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda()
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.D(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.opt.LAMBDA
        return gradient_penalty

    def forward(self,latent_y,ebd_y):
        micro_patches = self.G(latent_y,ebd_y)
        self.ebd_y = ebd_y
        self.macro_patcher = self.macro_from_micro(micro_patches)
        
    def backward(self,x,y):
        #update D()
        x = x.cuda()
        self.D.zero_grad()
        macro_patcher = self.macro_patcher.detach()
        fakeD,_ = self.D(macro_patcher,self.ebd_y)#y有问题！
        realD,realDH = self.D(x.cuda(),y)#y有问题！
        gradient_penalty = self.calc_gradient_penalty(realD,fakeD)
        d_loss = fakeD.mean()-realD.mean()+gradient_penalty+self.opt.ALPHA*self.Lsloss(realDH,y)
        d_loss.backward()
        self.optimizerD.step()
        self.d_losses.append(d_loss.item())
        #update G()
        self.G.zero_grad()
        realG,realGH = self.D(self.macro_patcher,self.ebd_y)#y有问题!
        g_loss = -realG.mean()+self.opt.ALPHA*self.Lsloss(realGH,self.ebd_y)
        g_loss.backward()
        self.optimizerG.step()
        self.g_losses.append(g_loss.item())
    
    def show_img(self):
        pass
    def show_loss(self):
        pass
        






# if __name__ == "__main__":
#     opt = option.Option()
#     y = torch.arange(opt.micro_in_macro)
#     g = GetLatentY(opt)
#     g.get_new_latent()
#     latent_y,ebd_y = g.get_latent_y(y)
#     print(latent_y.size())
#     print(ebd_y.size())
#     print(ebd_y)
#     print(ebd_y.repeat((opt.batchsize,1)))
#     print(ebd_y.size())

