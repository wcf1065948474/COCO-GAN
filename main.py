import train
import dataset
import option
import torch
import train
import models

opt = option.Option()
celeba_dataset = dataset.MacroCelebaDataset(opt)
dataloader = torch.utils.data.DataLoader(celeba_dataset,opt.batchsize,shuffle=True,num_workers=4)
gan = train.COCOGAN(opt)
latent_generator = train.GetLatentY(opt)
pos_list = models.GeneratePosList(opt)


for e in range(opt.epoch):
    for real_macro_list in dataloader:
        latent_generator.get_new_latent()
        for pos,real_macro in enumerate(real_macro_list):
            latent_y,micro_ebdy = latent_generator.get_latent_y(pos_list.get_pos_list(pos,False))
            gan.forward(latent_y,micro_ebdy)
            macro_ebdy = latent_generator.get_ebd(pos_list.get_pos_list(pos))
            gan.backward(real_macro,macro_ebdy)
