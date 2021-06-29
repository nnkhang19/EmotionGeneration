import torch 
import torch.nn as nn 


class SAdaIN(nn.Module):
    """
        @ref: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
    """
    def __init__(self, latent_size, channels):
        super(SAdaIN, self).__init__()
        self.linear = nn.Linear(latent_size, channels * 2)

    def forward(self, x, latent):
        style = self.linear(latent)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)    # [batch_size, 2, n_channels, ...]
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x


def test():
	
	in_channels = 3
	img_size=256

	# x is source image 
	x = torch.randn((2, in_channels, img_size, img_size))


	# y is target image
	y = torch.rand((2, in_channels))
	#y = torch.randn((2, in_channels, img_size, img_size))

	ada = SAdaIN(in_channels, in_channels)

	t = ada(x, y)

	print("Image size: ", x.shape)
	print("Style size: ", y.shape)
	print("Result size: " ,t.shape)


if __name__ == '__main__':
	test()