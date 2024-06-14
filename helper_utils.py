import os
# import rasterio
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


def to_negative(input_path, output_path):
    img = cv2.imread(input_path, 1)
    img = 255 - img
    cv2.imwrite(output_path, img)    

    
def to_gray_scale(filename, out):
    img = Image.open(filename)
    # Convert image to grayscale
    gray_img = np.array(img.convert('L'))
    cv2.imwrite(f'{out}.tif', gray_img)


def bg_to_gray(input_path, output_path):
    img = Image.open(input_path).convert('L')
    pixels = img.getdata()

    new_pixels = [0]* len(pixels)
    for i, pixel in enumerate(pixels):
        if pixel < 40:
            new_pixels[i] = 115
        else:
            new_pixels[i] = pixel

    img.putdata(new_pixels)
    img.save(output_path)




# to_gray_scale('data/CS_neurons/input/38_y.png', 'lol')
# to_gray_scale('data/DIC-C2DH-HeLa/01/t000.tif')

# to_negative('data/t024.tif')
# img = Image.open('data/t024.tif')
# data = img.load()
# print(data[0, 0])

# img_to_tif('data/neuron', format='jpg')
to_gray_scale('data/neuron.jpg', 'data/neuron')
bg_to_gray('data/neuron.tif', 'data/neuron.tif')


# Call the function with the path to your tif file
#plot_tifs('data/DIC-C2DH-HeLa/01/', 'segmentation')
# img1 = read_input_tif('data/Fluo-N2DL-HeLa/01/t012.tif')
# fig1 = px.imshow(img1)
# fig1.show()