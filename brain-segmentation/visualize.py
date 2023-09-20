from typing import List, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def fig2data(fig):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    return np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)


def create_image_visual(
        source: np.ndarray,
        target: Optional[np.ndarray] = None,
        output: Optional[np.ndarray] = None,
        title: str = ''
) -> np.ndarray:
    index = source.squeeze().shape[-1] // 2

    source = source.squeeze()[..., index]

    if target is not None:
        target = target.squeeze()[..., index]

    if output is not None:
        output = output.squeeze()[..., index]

    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(131)
    ax1.imshow(source, cmap='gray')
    plt.title('input')

    if output is not None:
        ax2 = fig.add_subplot(132)
        ax2.imshow(output)
        plt.title('prediction')

    if target is not None:
        ax3 = fig.add_subplot(133)
        ax3.imshow(target)
        plt.title('target')

    if title:
        plt.suptitle(title)

    image = fig2data(fig)
    # plt.savefig('{}.png'.format(title))
    return image
