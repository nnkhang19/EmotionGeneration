import matplotlib.pyplot as plt
import torch  
import os 
import config as cf 
import matplotlib.pyplot as plt 


def show_images(display_list):
    title = ['Identity', 'Attribute', 'Generated Image']
    for i in range(len(display_list)):
      plt.subplot(1, 3, i+1)
      plt.title(title[i])
      # getting the pixel values between [0, 1] to plot it.
      plt.imshow(display_list[i].permute(1,2,0))
      plt.axis('off')
    plt.show()

def convert_to_image(tensor, std, mean):
    return tensor * std[:, None, None] + mean[:, None, None]


def save_checkpoint(item, item_name):
    assert item_name in ['generator', 'discriminator', 'encoder', 'optimizer']

    path = os.path.join('saved_models', item_name + '.pth')
    print("Saving checkpoint----------")
    torch.save(item.state_dict(), path)

def load_checkpoint(item_name):
    
    assert item_name in ['generator', 'discriminator', 'encoder', 'optimizer']

    path = os.path.join('saved_models', item_name + '.pth')

    return toch.load(path)



if __name__ == '__main__':
    item_name = 'generator'
    epoch = 1
    path = os.path.join('saved_models', item_name + f'_{epoch}.pth')
    print(path)