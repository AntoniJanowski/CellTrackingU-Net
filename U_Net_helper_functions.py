import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
import seaborn as sns

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

def validate_model(model, val_dataset, device = torch.device('cuda')):
    model.eval()
    with torch.no_grad():
        length = len(val_dataset)
        model = model
        total_loss = 0
        for i in range(length):
            data, labels = val_dataset[i]
            data = data.to(device)
            labels = labels.to(device)
            labels = labels.long()
            output = model(data.unsqueeze(dim = 0))
            loss = model.loss_fn(output, labels)
            total_loss += loss.item()
        return total_loss / length  

def process_images(image_paths, label_paths, output_folder_image, output_folder_label, k=1):
    """
    Processes each image in image_paths by randomly cropping, rotating, and saving k times.

    Args:
        image_paths (list of str): List of paths to the input images.
        label_paths (list of str): List of paths to the input labels.
        output_folder (str): Path to the folder where processed images will be saved.
        k (int): Number of times to process each image.
    """
    if not os.path.exists(output_folder_image):
        raise Exception(f"{output_folder_image} No such file or directory") 
        # os.makedirs(output_folder)
    
    for i in range(len(image_paths)):
        image = Image.open(image_paths[i])
        label = Image.open(label_paths[i])
        image_name = os.path.basename(image_paths[i])
        label_name = os.path.basename(label_paths[i])
        for i in range(k):
            cropped_image, cropped_label = random_crop(image, label, (512, 512))
            rotated_image, rotated_label = random_rotate(cropped_image, cropped_label)
            output_path_image = os.path.join(output_folder_image, f"{os.path.splitext(image_name)[0]}_{i}.tif")
            output_path_label = os.path.join(output_folder_label, f"{os.path.splitext(label_name)[0]}_{i}.tif")
            rotated_image.save(output_path_image, format='TIFF')
            rotated_label.save(output_path_label, format='TIFF')

def random_crop(image, label, size):
    """
    Randomly crops the image to the given size.

    Args:
        image (PIL.Image): The input image.
        size (tuple): The size of the crop (width, height).

    Returns:
        PIL.Image: The cropped image.
    """
    width, height = image.size
    new_width, new_height = size
    if width < new_width or height < new_height:
        raise ValueError("Crop size is larger than the image size.")
    
    left = random.randint(0, width - new_width)
    top = random.randint(0, height - new_height)
    right = left + new_width
    bottom = top + new_height

    return image.crop((left, top, right, bottom)), label.crop((left, top, right, bottom))

def random_rotate(image, label, fillcolor = 32999, fillcolor_label = 0):
    """
    Randomly rotates the image by a random angle.

    Args:
        image (PIL.Image): The input image.

    Returns:
        PIL.Image: The rotated image.
    """
    angle = random.uniform(0, 360)
    return image.rotate(angle, expand=False, fillcolor = fillcolor), label.rotate(angle, expand=False, fillcolor = fillcolor_label)


def plot_results(arr1, arr2, arr3):
    """
    This function takes three NumPy arrays and displays their heatmaps in a single row.
    
    Parameters:
    arr1 (numpy.ndarray): First array.
    arr2 (numpy.ndarray): Second array.
    arr3 (numpy.ndarray): Third array.
    """
    
    # Setting up the figure and axes
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    min_val = min(arr1.min(), arr2.min(), arr3.min())
    max_val = max(arr1.max(), arr2.max(), arr3.max())
    
    # Creating heatmaps
    sns.heatmap(arr1, ax=axs[0], cbar=False, cmap='viridis')
    axs[0].set_title('Orginal image')
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    
    sns.heatmap(arr2, ax=axs[1], cbar=False, cmap='viridis', vmin=min_val, vmax=max_val)
    axs[1].set_title('Label')
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    
    sns.heatmap(arr3, ax=axs[2], cbar=False,cmap='viridis', vmin=min_val, vmax=max_val)
    axs[2].set_title('Model prediction')
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    
    # Adjusting the layout
    plt.tight_layout()
    
    # Displaying the heatmaps
    plt.show()

def pixel_wise_accuracy(tensor1, tensor2):
    """
    Calculate the pixel-wise accuracy between two tensors.
    
    Parameters:
    tensor1 (torch.Tensor): First tensor.
    tensor2 (torch.Tensor): Second tensor.
    
    Returns:
    float: Pixel-wise accuracy.
    """
    # Ensure the tensors have the same shape
    assert tensor1.shape == tensor2.shape, "Tensors must have the same shape"
    
    # Calculate the number of matching elements
    matching_elements = torch.sum(tensor1 == tensor2).item()
    
    # Calculate the total number of elements
    total_elements = tensor1.numel()
    
    # Calculate pixel-wise accuracy
    accuracy = matching_elements / total_elements
    
    return accuracy

def true_positive_rate(predictions, ground_truth):
    """
    Calculate the true positive rate (TPR) between two binary tensors.
    
    Parameters:
    predictions (torch.Tensor): Predicted binary tensor.
    ground_truth (torch.Tensor): Ground truth binary tensor.
    
    Returns:
    float: True positive rate (TPR).
    """
    # Ensure the tensors have the same shape
    assert predictions.shape == ground_truth.shape, "Tensors must have the same shape"
    
    # Calculate true positives and false negatives
    true_positives = torch.sum((predictions == 1) & (ground_truth == 1)).item()
    false_negatives = torch.sum((predictions == 0) & (ground_truth == 1)).item()

    # Avoid division by zero
    if true_positives + false_negatives == 0:
        return 0.0
    
    # Calculate TPR
    tpr = true_positives / (true_positives + false_negatives)
    
    return tpr

def create_folder_if_not_exists(path, folder_name):
    folder_path = os.path.join(path, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def perform_data_augmentation_on_dataset(path_to_data_folder, dataset_name, k):
    """
    Performs data augmentation on a given dataset, randomly cropping and rotating the images within, repeating it k times for each image
    args:
        -path_to_data_folder: Path to a folder with extracted datasets
        -dataset_name: name of the dataset to agument
        -k: Number of copies of each image from the dataset
    """
    create_folder_if_not_exists(path_to_data_folder + dataset_name, 'augmented_images')
    create_folder_if_not_exists(path_to_data_folder + dataset_name, 'augmented_image_labels')

    paths_01 =  list_files_in_folder(convert_backslashes_to_forward_slashes(path_to_data_folder + dataset_name + dataset_name + '/01'))
    paths_01_labels =  list_files_in_folder(convert_backslashes_to_forward_slashes(path_to_data_folder + dataset_name + dataset_name + '/01_ST'))
    paths_02 =  list_files_in_folder(convert_backslashes_to_forward_slashes(path_to_data_folder + dataset_name + dataset_name + '/02'))
    paths_02_labels =  list_files_in_folder(convert_backslashes_to_forward_slashes(path_to_data_folder + dataset_name + dataset_name + '/02_ST'))

    image_paths = paths_01 + paths_02
    label_paths = paths_01_labels + paths_02_labels

    output_folder_image = convert_backslashes_to_forward_slashes(path_to_data_folder + dataset_name + r"/augmented_images")
    output_folder_label = convert_backslashes_to_forward_slashes(path_to_data_folder + dataset_name + r"/augmented_image_labels")
    process_images(image_paths, label_paths, output_folder_image, output_folder_label, k=k)