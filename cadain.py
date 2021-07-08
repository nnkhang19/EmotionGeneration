
import torch
import torch.nn as nn

class CAdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def mu(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the average across
        it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
        return torch.sum(x,(2,3))/(x.shape[2]*x.shape[3])

    def sigma(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
        across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
        the permutations are required for broadcasting"""
        return torch.sqrt((torch.sum((x.permute([2,3,0,1])-self.mu(x)).permute([2,3,0,1])**2,(2,3))+0.000000023)/(x.shape[2]*x.shape[3]))

    def forward(self, x, y):
        """ Takes a content embeding x and a style embeding y and changes
        transforms the mean and standard deviation of the content embedding to
        that of the style. [See eq. 8 of paper] Note the permutations are
        required for broadcasting"""
        return (self.sigma(y)*((x.permute([2,3,0,1])-self.mu(x))/self.sigma(x)) + self.mu(y)).permute([2,3,0,1])


def test():
    in_channels = 512
    img_size=256

    # x is source image 
    x = torch.randn((2, in_channels, img_size, img_size))


    # y is target image
    #y = torch.rand((2, in_channels))
    y = torch.randn((2, in_channels, 16, 16))

    ada = CAdaIN()

    t = ada(x, y)

    print("Image size: ", x.shape)
    print("Style size: ", y.shape)
    print("Result size: " ,t.shape)


if __name__ == '__main__':
    test()


    