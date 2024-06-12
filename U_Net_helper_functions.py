import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_heatmap(data):
    """
    Plot a heatmap from a 2D NumPy array.

    :param data: 2D NumPy array representing the heatmap data.
    """
    plt.imshow(data, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.show()

def convert_backslashes_to_forward_slashes(path):
    """
    Converts all backslashes in the given path to forward slashes.

    :param path: The file path as a string.
    :return: The modified file path with forward slashes.
    """
    return path.replace('\\', '/')

def list_files_in_folder(folder_path):
    """
    Returns a list of paths to all files in the given folder.

    :param folder_path: Path to the folder.
    :return: List of file paths.
    """
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def channel_comparison(tensor):
    """
    Compare the two channels of the input tensor and return a numpy array with 1 where the second channel
    has a greater value than the first channel and 0 otherwise.

    Args:
    tensor (torch.Tensor): Input tensor of shape (1, 2, 512, 512).

    Returns:
    numpy.ndarray: Output array of shape (512, 512) with 1s and 0s.
    """
    if tensor.shape != (1, 2, 512, 512):
        raise ValueError("Input tensor must have shape (1, 2, 512, 512)")

    # Extract the first and second channels
    first_channel = tensor[0, 0, :, :]
    second_channel = tensor[0, 1, :, :]

    # Compare the values
    comparison_result = (second_channel > first_channel).numpy().astype(np.uint8)

    return comparison_result

def Train_Unet(train_dataloader, epochs, model,
                optimizer = None,
                loss_fn = torch.nn.CrossEntropyLoss()):
    model = model
    if optimizer == None:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    for epoch in range(epochs):
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            labels = labels.squeeze().long()
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()
            print(f'BATCH: {i}, LOSS: {loss}')
        print(f'End of epoch {epoch}')
    return model