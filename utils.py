import matplotlib.pyplot as plt
import torch  


def show_image(tensor, index):
    if torch.is_tensor(tensor):
        image = tensor[index].detach()
    else:
        image = tensor
    plt.imshow(image.permute(1,2,0))
    plt.xticks([])
    plt.yticks([])

def convert_to_image(tensor, std, mean):
    return tensor * std[:, None, None] + mean[:, None, None]


def save_checkpoint(path, model,learning_rate, optimizer):
    print("Saving checkpoint----------")
    torch.save({
            'learning_rate' : learning_rate,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),

        }, path

    )

def load_checkpoint(path):
    checkpoint = torch.load(path)
    
    return checkpoint



