import os
import rasterio
import plotly.graph_objects as go

def read_tif(file_path):
    # OUr tiffs have only one layer.
    with rasterio.open(file_path) as src:
        return src.read(1)

def plot_tifs(directory):
    frames = []
    for file in os.listdir(directory):
        if file.endswith(".tif"):
            data = read_tif(os.path.join(directory, file))
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
# plot_tifs('DIC-C2DH-HeLa/01_ST/SEG/')
