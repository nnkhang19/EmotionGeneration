import torch
#import albumentations as A  
#from albumentations.pytorch import ToTensorV2
import torchvision

root_dir='/content/EmotionGeneration/data256x256'
device= "cuda" if torch.cuda.is_available() else "cpu"
epochs = 2
lr = 2e-4
batch_size = 2
lambda_identity = 10
lambda_gp=1e-5
lambda_fm=10 
num_workers=2
pin_memory = True

load_model = False 
save_model = True 

gen_checkpoint='gen.pth.tar'
disc_checkpoint='disc.pth.tar'
enc_checkpoint='enc.pth.tar'

'''
transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5), 
        A.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], max_pixel_value=255),
    ],
        ToTensorV2(),
    additional_targets={'image0':'image'}
) 
'''
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

