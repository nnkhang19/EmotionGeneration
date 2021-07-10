import os
import torch
import pandas as pd
import numpy as np
import random
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class CelebA(Dataset):
	def __init__(self, root_dir, transforms=None):
		self.root = root_dir
		self.transforms=transforms
		self.files = os.listdir(self.root)

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		image_name = os.path.join(self.root, self.files[idx])
		image = io.imread(image_name)
		
		t_idx = [random.randint(0, len(self.files)-1) for i in range(len(idx))] if type(idx) == 'list' else random.randint(0, len(self.files)-1)

		while t_idx == idx:
			t_idx = [random.randint(0, len(self.files)-1) for i in range(len(idx))] if type(idx) == 'list' else random.randint(0, len(self.files)-1)

		target_name=os.path.join(self.root, self.files[t_idx])
		target=io.imread(target_name)

		if self.transforms:
			image = self.transforms(image)
			target=self.transforms(target)
		
		return image, target