import train
import dataset
import option
import torch
import models

opt = option.Option()
celeba_dataset = dataset.MacroCelebaDataset(opt)
dataloader = torch.utils.data.DataLoader(celeba_dataset,opt.batchsize,shuffle=True,num_workers=1,drop_last=True)
gan = train.COCOGAN(opt)
latent_generator = train.GetLatentY(opt)
pos_list = models.GeneratePosList(opt)


# for e in range(opt.epoch):
#     for real_macro_list in dataloader:
#         latent_generator.get_new_latent()
#         for pos,real_macro in enumerate(real_macro_list):
#             latent_y,micro_ebdy = latent_generator.get_latent_y(pos_list.get_pos_list(pos,False))
#             gan.forward(latent_y,pos_list.get_pos_list(pos,False))
#             macro_ebdy = latent_generator.get_ebd(pos_list.get_pos_list(pos))
#             gan.backward(real_macro,macro_ebdy)


show = False
for e in range(0,100):
    show = True
    for real_macro_list in dataloader:
        gan.latent_ebdy_generator.get_latent()
        for pos,real_macro in enumerate(real_macro_list):
            latent_y,micro_ebdy = gan.latent_ebdy_generator.get_latent_ebdy(pos)
            gan.forward(latent_y,latent_y)
            macro_ebdy = gan.latent_ebdy_generator.get_ebdy(pos,'macro')
            gan.backward(real_macro,macro_ebdy)
            if show:
                gan.show_img(pos)
                if pos == 8:
                    show = False
    print(e)
    gan.show_loss()
    gan.reset_loss()
    gan.save_network(e)            