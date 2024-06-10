import os
import rasterio
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

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

# Call the function with the path to your tif file
#plot_tifs('data/DIC-C2DH-HeLa/01/', 'segmentation')

# img = read_seq_tif('data/DIC-C2DH-HeLa/01_ST/SEG/man_seg000.tif')
# fig = px.imshow(img)
# fig.show()