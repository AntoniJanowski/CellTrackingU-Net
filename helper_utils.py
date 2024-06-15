import os
import rasterio
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def read_seq_tif(file_path):
    # OUr tiffs have only one layer.
    with rasterio.open(file_path) as src:
        img = src.read()
        assert img.shape[0] == 1, f'Tif has more than one layer. Number of layer = {img.shape[0]}'
        img = np.where(img != 0, 1, img)
        return img[0]
    
def read_input_tif(file_path):
    # OUr tiffs have only one layer.
    with rasterio.open(file_path) as src:
        img = src.read()
        assert img.shape[0] == 1, f'Tif has more than one layer. Number of layer = {img.shape[0]}'
        return img[0]

def plot_tifs(directory, mode):
    frames = []
    for file in os.listdir(directory):
        if file.endswith(".tif"):
            if mode == 'input':
                data = read_input_tif(os.path.join(directory, file))
            elif mode == 'segmentation':
                data = read_seq_tif(os.path.join(directory, file))
            assert mode == 'input' or mode == 'segmentation', f'Mode have to one of 2 options: "input" or "segmentation". Got: {mode}'
            frames.append(go.Frame(data=[go.Heatmap(z=data)]))

    fig = go.Figure(
        data=[go.Heatmap(z=frames[0]['data'][0]['z'])],
        layout=go.Layout(
            updatemenus=[dict(type="buttons",
                              buttons=[dict(label="Play",
                                            method="animate",
                                            args=[None])])]),
        frames=frames
    )
    fig.show()


def img_to_tif(prefix, input_format='png'):
    img = cv2.imread(f'{prefix}.{input_format}')
    cv2.imwrite(f'{prefix}.tif', img)


def img_transform(filename, output=None, light_bg=False):
    # to gray scale
    img = Image.open(filename)
    img = np.array(img.convert('L'))

    # negative
    if light_bg:
        img = 255 - img
    
    # bg to gray
    img = np.where(img < 40, img + 115, img)

    if not output:
        return img
    
    cv2.imwrite(output, img)


# img_transform('data/CS_neurons/input/38_y.png', output='neurons.tif')
# img_transform('data/CS_BCCD/input/0a3b53c7-e7ab-4135-80aa-fd2079d727d6.jpg', output='BCCD.tif', light_bg=True)
# img_transform('data/CS_blood/input/2_DAPI.tif', output='blood.tif')
