import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
import cv2
from skimage import io, transform
import config as cf
from generator import Generator
from discriminator import VggDiscriminator, StyleDiscriminator
from encoder import VggEncoder, StyleEncoder
from dataset import CelebA
from loss_function import original_gan_loss, w_gan_loss, compute_gradient_penalty, identity_loss, feature_match_loss
from utils import show_images, save_checkpoint, load_checkpoint, convert_to_image



#writer=SummaryWriter()

def train_one_epoch(enc, gen, disc, image_loader, optim):
    loop = tqdm(image_loader, leave = True)
    
    for idx, (image, target) in enumerate(loop):
        image = image.to(cf.device)
        target = target.to(cf.device)


        encoded_maps=enc(target)
        fake=gen(image, encoded_maps[-1])
        maps1, classes1= disc(fake)
        maps2, classes2=disc(target)
        
        loss_d = w_gan_loss(fake, image, mode='D') 
        loss_enc= cf.lambda_fm *  0.5 * (feature_match_loss(encoded_maps, maps2) + feature_match_loss(encoded_maps, maps1))
        loss_g = w_gan_loss(fake, mode='G') + cf.lambda_identity * identity_loss(image, fake)

        total_loss = loss_d + loss_enc + loss_g

        optim.zero_grad()
        total_loss.backward()
        optim.step()
        
        '''
        optim_disc.zero_grad()    
        loss_d.backward()
        optim_disc.step()
    
        optim_gen.zero_grad()
        loss_g.backward()
        optim_gen.step()

        optim_enc.zero_grad()
        loss_enc.backward()
        optim_enc.step()   
        '''
           

        if idx % 200 == 0:
            image = image.detach().cpu()
            fake = fake.detach().cpu()
            target = target.detach().cpu()
            image = convert_to_image(image, cf.std, cf.mean)
            fake  = convert_to_image(fake, cf.std, cf.mean)
            target = convert_to_image(target, cf.std, cf.mean)

            '''
            display_list = [image[0], target[0], fake[0]]
            title = ['Identity', 'Attribute', 'Generated Image']
            for i in range(len(display_list)):
              plt.subplot(1, 3, i+1)
              plt.title(title[i])
              # getting the pixel values between [0, 1] to plot it.
              plt.imshow(display_list[i].permute(1,2,0))
              plt.axis('off')
            plt.show()
            '''

            save_image(image[0], "saved_images/identity_{}.png".format(idx))
            save_image(fake[0], "saved_images/generated_{}.png".format(idx))
            save_image(target[0], "saved_images/attribute_{}.png".format(idx))
        

def main():
    gen = Generator(image_channels=3)
    #disc = Discriminator(requires_grad=True,pretrained=False)
    #enc = Encoder(requires_grad=True,pretrained=False)
    disc = StyleDiscriminator()
    enc = StyleEncoder()

    gen.to(cf.device)
    disc.to(cf.device)
    enc.to(cf.device)

    opt = optim.Adam(
            list(disc.parameters()) + list(gen.parameters()) + list(enc.parameters()),
            lr=cf.lr, betas=(0.5,0.999)
        )
    '''
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
    '''

    if cf.load_model:
        gen = load_checkpoint('generator')
        disc = load_checkpoint('discriminator')
        enc = load_checkpoint('encoder')
        opt = load_checkpoint('optimizer')
        

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
        enc.train(mode=True)
        gen.train(mode=True)
        train_one_epoch(enc, gen, disc, image_loader, opt)
        
        if cf.save_model:
            save_checkpoint(gen, 'generator')
            save_checkpoint(disc, 'discriminator')
            save_checkpoint(enc, 'encoder')
            save_checkpoint(opt, 'optimizer')
          
        sample_identity = io.imread('identity.jpg')
        sample_attribute = io.imread('attribute.jpg')

        image = cf.transform(sample_identity).unsqueeze(0).to(device)
        target = cf.transform(sample_attribute).unsqueeze(0).to(device)
        
        enc.train(mode=False)
        gen.train(mode=False)
        encoded_maps=enc(target)
        fake=gen(image, encoded_maps[-1])
        fake= fake.cpu()

        fake = convert_to_image(fake, cf.std, cf.mean)

        cv2_imshow(fake.permute(1,2,0))






if __name__ == '__main__':
    main()





