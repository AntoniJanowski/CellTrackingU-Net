import torch
import numpy as np
import cv2
import scipy.ndimage


def get_average_prediction(image_path, model_path):

    model = torch.load(model_path)
    model.eval()
    image = cv2.imread(image_path)

    predictions = []

    # Create transformations
    rotations = [0, 90, 180, 270]
    for rotation in rotations:
        # Rotate the image
        rotated_image = scipy.ndimage.rotate(image, rotation)
        tensor = torch.from_numpy(rotated_image).float().unsqueeze(0)
        prediction = model(tensor)
        # Rotate back to original
        rotated_back_prediction = scipy.ndimage.rotate(prediction.detach().numpy(), -rotation)
        # Append to the list
        predictions.append(rotated_back_prediction)

        # Do the same with a symmetrical image
        symmetrical_image = np.fliplr(rotated_image)
        tensor = torch.from_numpy(symmetrical_image).float().unsqueeze(0)
        prediction = model(tensor)
        flipped_back_prediction = np.fliplr(scipy.ndimage.rotate(prediction.detach().numpy(), -rotation))
        predictions.append(flipped_back_prediction)

    average_prediction = np.mean(predictions, axis=0)
    return average_prediction

