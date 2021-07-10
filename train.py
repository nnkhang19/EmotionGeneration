import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm
from torch.utils.data import DataLoader
import config as cf
from generator import Generator
from discriminator import VggDiscriminator, StyleDiscriminator
from encoder import VggEncoder, StyleEncoder
from dataset import CelebA
from loss_function import original_gan_loss, w_gan_loss, compute_gradient_penalty, identity_loss, feature_match_loss
from utils import show_image, save_checkpoint, load_checkpoint, convert_to_image

writer=SummaryWriter()

def train_one_epoch(enc, gen, disc, image_loader, optim_enc, optim_gen, optim_disc):
    loop = tqdm(image_loader, leave = True)
    
    for idx, (image, target) in enumerate(loop):
        image = image.to(cf.device)
        target = target.to(cf.device)


        encoded_maps=enc(target)
        fake=gen(image, encoded_maps[-1])
        maps1, classes1= disc(fake)
        maps2, classes2=disc(image)
        
        loss_d = w_gan_loss(fake, image, mode='D') + cf.lambda_gp * compute_gradient_penalty(disc, image, fake) + cf.lambda_fm * feature_match_loss(maps1, maps2)
        loss_enc= cf.lambda_fm * 0.5 * (feature_match_loss(encoded_maps, maps2) + feature_match_loss(encoded_maps, maps1))
        loss_g = w_gan_loss(fake, mode='G') + cf.lambda_identity * identity_loss(image, fake)

        optim_disc.zero_grad()    
        loss_d.backward()
        optim_disc.step()
    
        optim_gen.zero_grad()
        loss_g.backward()
        optim_gen.step()

        optim_enc.zero_grad()
        loss_enc.backward()
        optim_enc.step()        

        if idx % 200 == 0:
            convert_to_image(image, cf.std, cf.mean)
            convert_to_image(fake, cf.std, cf.mean)
            show_image(image, 0)
            show_image(fake, 0)
            save_image(image[0], "saved_images/image_{}.png".format(idx))
            save_image(fake[0], "saved_images/fake_{}.png".format(idx))

def main():
    gen = Generator(image_channels=3)
    #disc = Discriminator(requires_grad=True,pretrained=False)
    #enc = Encoder(requires_grad=True,pretrained=False)
    disc = StyleDiscriminator()
    enc = StyleEncoder()

    gen.to(cf.device)
    disc.to(cf.device)
    enc.to(cf.device)
    
    opt_disc = optim.Adam(
        list(disc.parameters()),
        lr = cf.lr, betas=(0.5,0.999),
    )
    opt_gen = optim.Adam(
        list(gen.parameters()),
        lr=cf.lr, betas=(0.5, 0.999)
    )
    opt_enc = optim.Adam(
        list(enc.parameters()),
        lr=cf.lr, betas=(0.5, 0.999)
    )

    if cf.load_model:
        gen_checkpoint=load_checkpoint(cf.gen_checkpoint)
        disc_checkpoint=load_checkpoint(cf.disc_checkpoint)
        enc_checkpoint=load_checkpoint(cf.enc_checkpoint)

        gen.load_state_dict(gen_checkpoint['model_state_dict'])
        disc.load_state_dict(disc_checkpoint['model_state_dict'])
        enc.load_state_dict(enc_checkpoint['model_state_dict'])

        opt_gen.load_state_dict(gen_checkpoint['optimizer'])
        opt_disc.load_state_dict(disc_checkpoint['optimizer'])
        opt_enc.load_state_dict(enc_checkpoint['optimizer'])

        cf.lr= gen_checkpoint['learning_rate']

    dataset = CelebA(cf.root_dir, cf.transform)
    num_samples=len(dataset)
    image_loader = DataLoader(
    dataset,
    batch_size = cf.batch_size,
    shuffle = True,
    num_workers = cf.num_workers,
    pin_memory = cf.pin_memory,

    )

    for e in range(cf.epochs):
        train_one_epoch(enc, gen, disc, image_loader, opt_enc, opt_gen, opt_disc)

        if cf.save_model:
            save_checkpoint(cf.gen_checkpoint, gen, cf.lr, opt_gen)
            save_checkpoint(cf.disc_checkpoint, disc, cf.lr, opt_disc)
            save_checkpoint(cf.enc_checkpoint, enc, cf.lr, opt_enc)





if __name__ == '__main__':
    main()





