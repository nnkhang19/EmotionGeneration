import torch 
import torch.nn as nn 
from generator import Generator
from discriminator import Discriminator
from encoder import Encoder
from lossfunction import original_gan_loss, w_gan_loss, compute_gradient_penalty, identity_loss


image_channels= 3

identity = torch.randn((2,3,256,256))
attribute = torch.randn((2,3,256,256))

gen = Generator(image_channels=image_channels)
disc = Discriminator(requires_grad=True,pretrained=False)
enc = Encoder(requires_grad=True,pretrained=False)

y = enc(attribute)

fake =gen(identity, y[-1])

features, classified = disc(fake)
_, real_classification = disc(identity)

#loss_d = original_gan_loss(classified, is_real=False) + original_gan_loss(real_classification, is_real=True)

#loss_d = w_gan_loss(fake, identity)
#loss_g = w_gan_loss(fake, mode='G')
recons = identity_loss(identity, fake)

#print(loss_d)
#print(loss_g)
print(recons)





