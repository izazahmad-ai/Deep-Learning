from torch import nn
import numpy as np
import os
import cv2


def zero_grad(model: nn.Module):
    for param in model.parameters():
        param.grad = None


def dice(x: np.ndarray, y: np.ndarray) -> float:
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class ImageSaver:
    def __init__(self, save_dir: str, save_name: str):
        self.save_dir = save_dir
        self.save_name = save_name
        self.path_to_save = os.path.join(save_dir, save_name)
        self.path_to_save_images = os.path.join(self.path_to_save, 'images')
        if not os.path.exists(self.path_to_save_images):
            os.makedirs(self.path_to_save_images)
        else:
            raise ValueError(f'ImageSaver: {self.path_to_save} already exists')

    def save_image(self, image: np.ndarray, name: int or str):
        assert len(image.shape) == 3
        name = str(name)

        cv2.imwrite(os.path.join(self.path_to_save_images, name + '.png'), image)

    def save_results(self, **kwargs):
        with open(os.path.join(self.path_to_save, 'results.txt'), 'w') as f:
            for key, value in kwargs.items():
                f.write(f'{key}: {value}\n')
