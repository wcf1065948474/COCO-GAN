import torch
import torch.nn as nn
import option
import models
import os
import numpy as np
import matplotlib.pyplot as plt

# class GetLatentY(object):
#     def __init__(self,opt):
#         self.opt = opt
#         self.ebd = torch.nn.Embedding(16,28)
#     def get_new_latent(self,isNew = False):
#         self.z = np.random.normal(0.0,1.0,(self.opt.batchsize,100)).astype(np.float32)
#         if isNew == False:
#             self.z = np.repeat(self.z,self.opt.micro_in_macro,0)
#         self.z = torch.from_numpy(self.z)
#     def get_ebd(self,y):
#         ebd_y = self.ebd(y)
#         ebd_y = ebd_y.expand(self.opt.batchsize,-1)
#         return ebd_y
#     def get_latent_y(self,y):
#         ebd_y = self.ebd(y)
#         ebd_y = ebd_y.repeat((self.opt.batchsize,1))
#         latent_y = torch.cat((self.z,ebd_y),1)
#         return latent_y,ebd_y

# class NewGetLatentY(object):
#     def __init__(self,opt):
#         self.opt = opt
#         self.ebd = torch.linspace(-1,1,steps=16,dtype=torch.float32)
#     def get_new_latent(self,isNew = False):
#         self.z = np.random.normal(0.0,1.0,(self.opt.batchsize,127)).astype(np.float32)
#         if isNew == False:
#             self.z = np.repeat(self.z,self.opt.micro_in_macro,0)
#         self.z = torch.from_numpy(self.z)
#     def get_ebd(self,y):
#         y = torch.LongTensor(y)
#         ebd_y = self.ebd(y).view(-1,1)
#         ebd_y = ebd_y.expand(self.opt.batchsize,-1)
#         return ebd_y
#     def get_latent_y(self,y):
#         y = torch.LongTensor(y)
#         ebd_y = self.ebd(y).view(-1,1)
#         ebd_y = ebd_y.repeat((self.opt.batchsize,1))
#         latent_y = torch.cat((self.z,ebd_y),1)
#         return latent_y,ebd_y


class Get_Latent_Y(object):
    def __init__(self,opt):
        self.opt = opt
        x = torch.linspace(-1,1,int(np.sqrt(opt.num_classes)))
        x = x.view(-1,1)
        x = x.expand(-1,int(np.sqrt(opt.num_classes)))
        y = torch.linspace(-1,1,int(np.sqrt(opt.num_classes)))
        y = y.view(1,-1)
        y = y.expand(int(np.sqrt(opt.num_classes)),-1)
        self.ebd = torch.stack((x,y),0)
        self.ebd = self.ebd.view(2,-1)
        self.ebd.requires_grad_()
        self.wh = int(np.sqrt(opt.num_classes))
        self.pos_table = torch.arange(opt.num_classes).view(self.wh,self.wh)
        self.max_area = int(np.sqrt(opt.num_classes)-np.sqrt(opt.micro_in_macro)+1)
        self.macro_table = self.pos_table[0:self.max_area,0:self.max_area]
        self.macro_table = self.macro_table.contiguous().view(-1)
    def get_latent(self):
        self.z = np.random.normal(0.0,1.0,(self.opt.batchsize,126)).astype(np.float32)
        self.z = np.tile(self.z,(self.opt.micro_in_macro,1))
        self.z = torch.from_numpy(self.z)

    def get_ebdy(self,pos,mode='micro'):
        if mode == 'micro':
            pos_list = []
            for i in range(int(np.sqrt(self.opt.micro_in_macro))):
                for j in range(int(np.sqrt(self.opt.micro_in_macro))):
                    pos_list.append(self.macro_table[pos]+i*self.wh+j)
            ebdy = self.ebd[:,pos_list]
            ebdy = torch.transpose(ebdy,0,1)
        else:
            ebdy = self.ebd[:,self.macro_table[pos]].view(2,1)
            ebdy = torch.transpose(ebdy,0,1)
        ebdy = np.repeat(ebdy,self.opt.batchsize,0)
        return ebdy

    def get_latent_ebdy(self,pos):
        ebdy = self.get_ebdy(pos)
        return torch.cat((self.z,ebdy),1),ebdy

class COCOGAN(object):
    def __init__(self,opt):
        self.opt = opt
        self.G = models.Generator(opt)
        self.D = models.Discriminator()
        self.Lsloss = torch.nn.MSELoss()
        self.optimizerG = torch.optim.Adam(self.G.parameters(),1e-4,(0,0.999))
        self.optimizerD = torch.optim.Adam(self.D.parameters(),4e-4,(0,0.999))
        self.d_losses = []
        self.g_losses = []
        self.G.cuda()
        self.D.cuda()
        self.G.apply(self.weights_init)
        self.D.apply(self.weights_init)
        self.latent_ebdy_generator = Get_Latent_Y(opt)
    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            if classname.find('Conditional') == -1:
                if m.weight is not None:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0)

    # def macro_from_micro(self,micro):
    #     hw = int(np.sqrt(self.opt.micro_in_macro))#macro每行每列有多少micro
    #     batchs = int(micro.size(0)/self.opt.micro_in_macro)#macro的batchs
    #     spatial = int(self.opt.micro_size*hw)#macro的分辨率
    #     macros = torch.empty((batchs,3,spatial,spatial),dtype=micro.dtype)
    #     for b in range(batchs):
    #         bb = b*self.opt.micro_in_macro
    #         for i in range(hw):
    #             ii = i*self.opt.micro_size
    #             for j in range(hw):
    #                 jj = j*self.opt.micro_size
    #                 macros[b,:,ii:ii+self.opt.micro_size,jj:jj+self.opt.micro_size] = micro[bb+hw*i+j].clone()
    #     return macros.cuda()

    def macro_from_micro(self,micro):
        microlist = []
        macrolist = []
        hw = int(np.sqrt(self.opt.micro_in_macro))
        for i in range(self.opt.micro_in_macro):
            microlist.append(micro[i*self.opt.batchsize:i*self.opt.batchsize+self.opt.batchsize])
        for j in range(hw):
            macrolist.append(torch.cat(microlist[j*hw:j*hw+hw],3))
        return torch.cat(macrolist,2)
    def calc_gradient_penalty(self,real_data,fake_data,ebd_y):
        #print real_data.size()
        alpha = torch.rand(self.opt.batchsize, 1,1,1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda()
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates,disc_h = self.D(interpolates,ebd_y)

        gradients = torch.autograd.grad(outputs=[disc_interpolates,disc_h], inputs=[interpolates,ebd_y],
                                    grad_outputs=[torch.ones(disc_interpolates.size()).cuda(),torch.ones(disc_h.size()).cuda()],
                                    create_graph=True, retain_graph=True, only_inputs=True)

        gradient_penalty = (gradients[0].norm(2,dim=1)-1)**2+((gradients[1].norm(2,dim=1)-1)**2).view(self.opt.batchsize,1,1)
        return gradient_penalty.mean()*self.opt.LAMBDA

    def forward(self,latent_ebdy,ebdy):
        latent_ebdy = latent_ebdy.cuda()
        ebdy = ebdy.cuda()
        micro_patches = self.G(latent_ebdy,ebdy)
        self.macro_patches = self.macro_from_micro(micro_patches)
    def forward_new(self,latent_y,y):
        assert y is list
        self.micro_patches = []
        tmp_patches = []
        hw = int(np.sqrt(self.opt.micro_in_macro))
        for y_i in y:
            y_i = y_i.cuda()
            latent_y = latent_y.cuda()
            self.micro_patches.append(self.G(latent_y,y_i))
        for i in range(hw):
            tmp_patches.append(torch.cat(self.micro_patches[i*hw:i*hw+hw],3))
        self.macro_patches = torch.cat(tmp_patches,2)

    def backward(self,x,ebd_y):
        #update D()
        x = x.cuda()
        ebd_y = ebd_y.cuda()
        self.D.zero_grad()
        self.macro_data = self.macro_patches.detach()
        macro_p = self.macro_data.cuda()
        fakeD,_ = self.D(macro_p,ebd_y)#y有问题！
        realD,realDH = self.D(x,ebd_y)#y有问题！
        gradient_penalty = self.calc_gradient_penalty(x,macro_p,ebd_y)
        d_loss = fakeD.mean()-realD.mean()+gradient_penalty+self.opt.ALPHA*self.Lsloss(realDH,ebd_y)
        d_loss.backward()
        self.optimizerD.step()
        self.d_losses.append(d_loss.item())
        #update G()
        self.G.zero_grad()
        realG,realGH = self.D(self.macro_patches,ebd_y)#y有问题!
        g_loss = -realG.mean()+self.opt.ALPHA*self.Lsloss(realGH,ebd_y)
        g_loss.backward()
        self.optimizerG.step()
        self.g_losses.append(g_loss.item())
    
    def generate_full_image_1(self):
        self.G.eval()
        z = np.random.normal(0.0,1.0,(1,127)).astype(np.float32)
        z = np.repeat(z,self.opt.num_classes,0)
        z = torch.from_numpy(z)
        y = torch.linspace(-1,1,self.opt.num_classes)
        y = y.view(self.opt.num_classes,1)
        latent_y = torch.cat((z,y),1)
        micros = self.G(latent_y,y)

        hw = int(np.sqrt(self.opt.micro_in_macro))#macro每行每列有多少micro
        spatial = int(self.opt.micro_size*hw)#macro的分辨率
        macros = torch.empty((spatial,spatial,3),dtype=micros.dtype)
        for h in range(int(np.sqrt(self.opt.num_classes))):
            for w in range(int(np.sqrt(self.opt.num_classes))):
                macros[h*self.opt.micro_size:h*self.opt.micro_size+self.opt.micro_size,w*self.opt.micro_size:w*self.opt.micro_size+self.opt.micro_size,:]=micros[h*int(np.sqrt(self.opt.num_classes))+w]
        plt.figure(figsize=(3,3))
        plt.axis('off')
        plt.imshow(macros)
        plt.show()
        self.G.train()
    def generate_full_image_2(self):
        pass

    def save_network(self,epoch_label):
        save_filename = "%s_netG.pth"%epoch_label
        save_path = os.path.join(self.opt.my_model_dir,save_filename)
        torch.save(self.G.state_dict(),save_path)
        save_filename = "%s_netD.pth"%epoch_label
        save_path = os.path.join(self.opt.my_model_dir,save_filename)
        torch.save(self.D.state_dict(),save_path)
    def load_network(self,epoch_label):
        filename = "%s_netG.pth"%epoch_label
        filepath = os.path.join(self.opt.my_model_dir,filename)
        self.G.load_state_dict(torch.load(filepath))
        filename = "%s_netD.pth"%epoch_label
        filepath = os.path.join(self.opt.my_model_dir,filename)
        self.D.load_state_dict(torch.load(filepath))

    def show_img(self):
        plt.figure(figsize=(3,3))
        plt.axis('off')
        img = self.macro_data.cpu()
        plt.imshow(img[0].permute(1,2,0))
        plt.show()
        
    def show_loss(self):
        avg_d_loss = sum(self.d_losses)/len(self.d_losses)
        avg_g_loss = sum(self.g_losses)/len(self.g_losses)
        print("d_loss={},g_loss={}".format(avg_d_loss,avg_g_loss))
    def reset_loss(self):
        self.d_losses = []
        self.g_losses = []
        



if __name__ == "__main__":
    opt = option.Option()
    opt.batchsize=2
    g = Get_Latent_Y(opt)
    g.get_latent()
    y = g.get_ebdy(5,'f')
    print(type(y))




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

