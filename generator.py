import torch 
import torch.nn as nn 
from blocks import Block
from self_attention import SelfAttention
from adain import SAdaIN
from cadain import CAdaIN



class Generator(nn.Module):
	def __init__(self, image_channels, features = 64):
		'''
			Kernel size, Stride, and padding are referenced from Pix2Pix architecture.

		'''
		super(Generator, self).__init__()
		self.down_block1 = nn.Sequential(
				nn.Conv2d(image_channels, features, 4, 2, 1, padding_mode = "reflect"),
				nn.LeakyReLU(0.2)
			) # 64
		
		self.down_block2 = Block(features, features*2, down= True, act = 'leaky')   #128
		self.down_block3 = Block(features*2, features*4, down= True, act = 'leaky') #256
		self.down_block4 = Block(features*4, features*8, down= True, act = 'leaky') #512
		self.down_block5 = Block(features*8, features*8, down= True, act = 'leaky') #512
		self.down_block6 = Block(features*8, features*8, down= True, act = 'leaky') #512
		self.down_block7 = Block(features*8, features*8, down= True, act = 'leaky') #512

		self.bottle_neck = nn.Sequential(
		    nn.Conv2d(features*8, features*8, 4, 2, 1), nn.ReLU()
		)

		self.adapt = CAdaIN()

		self.attention2  = SelfAttention(features*8)
		self.attention3  = SelfAttention(features*8)
		self.attention4  = SelfAttention(features*8)
		self.attention5  = SelfAttention(features*8)
		self.attention6  = SelfAttention(features*4)
		self.attention7  = SelfAttention(features*2)
		self.attention8  = SelfAttention(features)
 
		self.up_block1 = Block(features*8, features*8, down= False, act = 'relu', use_dropout= True)      #512
		self.up_block2 = Block(features*8, features*8, down= False, act = 'relu', use_dropout= True)      #512
		self.up_block3 = Block(features*8, features*8, down= False, act = 'relu', use_dropout= True)      #512
		self.up_block4 = Block(features*8, features*8, down= False, act = 'relu', use_dropout= False)     #512
		self.up_block5 = Block(features*8, features*4, down= False, act = 'relu', use_dropout= False)     #512
		self.up_block6 = Block(features*4, features*2, down= False, act = 'relu', use_dropout= False)     #512
		self.up_block7 = Block(features*2, features, down= False, act = 'relu', use_dropout= False)       #256

		self.output_block = nn.Sequential(
				nn.ConvTranspose2d(features, image_channels, kernel_size=4, stride=2, padding=1),
				nn.Tanh()
			)


	def forward(self, x, y):
		d1 = self.down_block1(x)
		d2 = self.down_block2(d1)
		d3 = self.down_block3(d2)
		d4 = self.down_block4(d3)
		d5 = self.down_block5(d4)
		d6 = self.down_block6(d5)
		d7 = self.down_block7(d6)
		#d7 = self.adapt(d7, y)

		bottle = self.bottle_neck(d7)

		bottle = self.adapt(bottle, y)
		u1 = self.up_block1(bottle)

		u2 = self.up_block2(self.attention2(u1,d7))
		u3 = self.up_block3(self.attention3(u2,d6))
		u4 = self.up_block4(self.attention4(u3,d5))
		u5 = self.up_block5(self.attention5(u4,d4))
		u6 = self.up_block6(self.attention6(u5,d3))
		u7 = self.up_block7(self.attention7(u6,d2))
		output = self.output_block(self.attention8(u7,d1))

		return output 


def test():
	in_channels = 3
	img_size=256

	# x is source image 
	x = torch.randn((2, in_channels, img_size, img_size))


	# y is target image
	#y = torch.rand((2, 100,))
	y = torch.randn((2, 512,))

	gen = Generator(in_channels)

	out = gen(x, y)

	print("Image size: ", x.shape)
	print("Style size: ", y.shape)
	print("Result size: " ,out.shape)

if __name__=='__main__':
	test()



