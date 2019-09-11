import torch
import torch.nn as nn
import option
import models
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# def plot_grad_flow(named_parameters):
#     '''Plots the gradients flowing through different layers in the net during training.
#     Can be used for checking for possible gradient vanishing / exploding problems.
    
#     Usage: Plug this function in Trainer class after loss.backwards() as 
#     "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
#     ave_grads = []
#     max_grads= []
#     layers = []
#     for n, p in named_parameters:
#         if(p.requires_grad) and ("bias" not in n):
#             layers.append(n)
#             ave_grads.append(p.grad.abs().mean())
#             max_grads.append(p.grad.abs().max())
#     plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
#     plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
#     plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
#     plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
#     plt.xlim(left=0, right=len(ave_grads))
#     plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
#     plt.xlabel("Layers")
#     plt.ylabel("average gradient")
#     plt.title("Gradient flow")
#     plt.grid(True)
#     plt.legend([Line2D([0], [0], color="c", lw=4),
#                 Line2D([0], [0], color="b", lw=4),
#                 Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
#     plt.show()

def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    # image_numpy = (image_numpy-np.min(image_numpy))/(np.max(image_numpy)-np.min(image_numpy))
    # image_numpy *= 255.0
    image_numpy = np.maximum(image_numpy, 0)
    image_numpy = np.minimum(image_numpy, 255)
    return image_numpy.astype(imtype)

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.show()

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
        macrolist = []
        hw = int(np.sqrt(self.opt.micro_in_macro))
        for j in range(hw):
            macrolist.append(torch.cat(micro[j*hw:j*hw+hw],3))
        return torch.cat(macrolist,2)
    def calc_gradient_penalty(self,real_data,fake_data,ebd_y):
        #print real_data.size()
        alpha = torch.rand(self.opt.batchsize, 1,1,1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()
        ebd_y.requires_grad_()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda()
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates,disc_h = self.D(interpolates,ebd_y)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=[torch.ones(disc_interpolates.size()).cuda(),torch.ones(disc_h.size()).cuda()],
                                    create_graph=True, retain_graph=True, only_inputs=True)

        gradient_penalty = (gradients[0].norm(2,dim=1)-1)**2
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

    def train(self,x,epoch,pos):
        latent_ebdy,_ = self.latent_ebdy_generator.get_latent_ebdy(pos)
        micro_patches = []
        for k in range(self.opt.micro_in_macro):
            tmp_latent = latent_ebdy[k*self.opt.batchsize:k*self.opt.batchsize+self.opt.batchsize].cuda()
            micro_patches.append(self.G(tmp_latent,tmp_latent))
        self.macro_patches = self.macro_from_micro(micro_patches)

        #update D()
        x = x.cuda()
        ebd_y = self.latent_ebdy_generator.get_ebdy(pos,'macro')
        ebd_y = ebd_y.cuda()
        self.D.zero_grad()
        self.macro_data = self.macro_patches.detach()
        fakeD,_ = self.D(self.macro_data,ebd_y)#y有问题！
        realD,realDH = self.D(x,ebd_y)#y有问题！
        gradient_penalty = self.calc_gradient_penalty(x,self.macro_data,ebd_y)
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
    def train_micro(self,x,epoch,pos):
        latent_ebdy,_ = self.latent_ebdy_generator.get_latent_ebdy(pos)
        latent_ebdy = latent_ebdy.cuda()
        self.micro_patches = self.G(latent_ebdy,latent_ebdy)
        #update D()
        x = x.cuda()
        ebd_y = self.latent_ebdy_generator.get_ebdy(pos,'macro')
        ebd_y = ebd_y.cuda()
        self.D.zero_grad()
        micro_data = self.micro_patches.detach()
        fakeD,fakeDH = self.D(micro_data,ebd_y)
        realD,realDH = self.D(x,ebd_y)
        gradient_penalty = self.calc_gradient_penalty(x,micro_data,ebd_y)
        d_loss = fakeD.mean()-realD.mean()+self.opt.ALPHA*(self.Lsloss(realDH,ebd_y)+self.Lsloss(fakeDH,ebd_y))+gradient_penalty
        d_loss.backward()
        self.optimizerD.step()
        self.d_losses.append(d_loss.item())
        #update G()
        self.G.zero_grad()
        realG,realGH = self.D(self.micro_patches,ebd_y)#y有问题!
        g_loss = -realG.mean()+self.opt.ALPHA*self.Lsloss(realGH,ebd_y)
        g_loss.backward()
        self.optimizerG.step()
        self.g_losses.append(g_loss.item())

    def generate_full(self):
        z = np.random.normal(0.0,1.0,(1,126)).astype(np.float32)
        z = np.repeat(z,self.opt.num_classes,0)
        z = torch.from_numpy(z)
        ebdy = torch.transpose(self.latent_ebdy_generator.ebd,0,1)
        latent_ebdy = torch.cat((z,ebdy),1)
        latent_ebdy = latent_ebdy.cuda()
        with torch.no_grad():
            micro_patches = self.G(latent_ebdy,latent_ebdy)
        micro_patches = torch.chunk(micro_patches,micro_patches.size(0),0)
        hw = int(np.sqrt(self.opt.num_classes))
        hlist = []
        for i in range(hw):
            hlist.append(torch.cat(micro_patches[i*hw:i*hw+hw],3))
        full_img = torch.cat(hlist,2)
        full_img = full_img[0]
        full_img = full_img.permute(1,2,0)
        plt.figure(figsize=(3,3))
        plt.axis('off')
        plt.imshow(full_img.cpu())
        plt.show()

    def generate_micro(self):
        z = np.random.normal(0.0,1.0,(self.opt.batchsize,126)).astype(np.float32)
        z = torch.from_numpy(z)
        ebdy = torch.transpose(self.latent_ebdy_generator.ebd,0,1)
        micro_list = []
        with torch.no_grad():
            for i in range(self.opt.num_classes):
                tmp_ebdy = ebdy[i].view(1,2)
                tmp_ebdy = tmp_ebdy.repeat(self.opt.batchsize,1)
                tmp_latent = torch.cat((z,tmp_ebdy),1).cuda()
                micro_list.append(self.G(tmp_latent,tmp_latent))
        hw = int(np.sqrt(self.opt.num_classes))
        hlist = []
        for i in range(hw):
            hlist.append(torch.cat(micro_list[i*hw:i*hw+hw],3))
        full_img = torch.cat(hlist,2)
        full_img = tensor2im(full_img)
        plt.figure(figsize=(3,3))
        plt.axis('off')
        plt.imshow(full_img)
        plt.show()

    
    # def generate_full_image_1(self):
    #     self.G.eval()
    #     z = np.random.normal(0.0,1.0,(1,127)).astype(np.float32)
    #     z = np.repeat(z,self.opt.num_classes,0)
    #     z = torch.from_numpy(z)
    #     y = torch.linspace(-1,1,self.opt.num_classes)
    #     y = y.view(self.opt.num_classes,1)
    #     latent_y = torch.cat((z,y),1)
    #     micros = self.G(latent_y,y)

    #     hw = int(np.sqrt(self.opt.micro_in_macro))#macro每行每列有多少micro
    #     spatial = int(self.opt.micro_size*hw)#macro的分辨率
    #     macros = torch.empty((spatial,spatial,3),dtype=micros.dtype)
    #     for h in range(int(np.sqrt(self.opt.num_classes))):
    #         for w in range(int(np.sqrt(self.opt.num_classes))):
    #             macros[h*self.opt.micro_size:h*self.opt.micro_size+self.opt.micro_size,w*self.opt.micro_size:w*self.opt.micro_size+self.opt.micro_size,:]=micros[h*int(np.sqrt(self.opt.num_classes))+w]
    #     plt.figure(figsize=(3,3))
    #     plt.axis('off')
    #     plt.imshow(macros)
    #     plt.show()
    #     self.G.train()
    # def generate_full_image_2(self):
    #     pass

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

    def show_img(self,pos):
        index = np.random.randint(0,self.opt.batchsize)
        if pos == 0:
            img = self.macro_data[index].cpu()
            self.macroA = img.permute(1,2,0)
        elif pos == 2:
            img = self.macro_data[index].cpu()
            self.macroB = img.permute(1,2,0)
        elif pos == 6:
            img = self.macro_data[index].cpu()
            self.macroC = img.permute(1,2,0)
        elif pos == 8:
            img = self.macro_data[index].cpu()
            self.macroD = img.permute(1,2,0)
            plt.figure(figsize=(3,3))
            plt.axis('off')
            tmp1 = torch.cat((self.macroA,self.macroB),1)
            tmp2 = torch.cat((self.macroC,self.macroD),1)
            tmp = torch.cat((tmp1,tmp2),0)
            plt.imshow(tmp)
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
