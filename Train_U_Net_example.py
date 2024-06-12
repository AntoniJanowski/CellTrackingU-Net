import sys
import os
import rasterio
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import torch
import pandas as pd
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import torch.nn as nn
from helper_utils import *
from U_Net_helper_functions import *
from U_Net import *
from TIF_dataset import *

paths_x =  list_files_in_folder(convert_backslashes_to_forward_slashes(r'C:\Users\Dell\Documents\Heidelberg_hackaton\CellTrackingU-Net\data\train\DIC-C2DH-HeLa\01'))
paths_y =  list_files_in_folder(convert_backslashes_to_forward_slashes(r'C:\Users\Dell\Documents\Heidelberg_hackaton\CellTrackingU-Net\data\train\DIC-C2DH-HeLa\01_ST'))

dataset = TIF_dataset(paths_x, paths_y)

sizes = [1, 8, 16, 32, 64, 128]
model = UNet(sizes)

train_dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss()

model = Train_Unet(train_dataloader, epochs = 2, model = model)
torch.save(model, 'example_UNET_model.pth')

#test that it works:
data, label = dataset[24]
output = model(data.unsqueeze(dim = 0))
model_return_numpy = model.generate_numpy_output_from_single_image(output)
print('Model output')
plot_heatmap(model_return_numpy)
print('Label')
plot_heatmap(label.squeeze().numpy())
print('original image')
plot_heatmap(data.squeeze().numpy())