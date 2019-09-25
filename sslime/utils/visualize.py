#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
This script contains some helpful functions to visualize the net and data
"""

import logging
import numpy as np
import os

logger = logging.getLogger(__name__)


def visualize_net(model, path):
    print(model)


# using matplotlib to plot images, it expects the data to be float and
# to be in range(0, 1)
def normalize_image_scale(image):
    image = image - np.min(image)
    image = image / (np.max(image) + np.finfo(np.float64).eps)
    return image


def visualize_image(
    image, path, name, order="CHW", normalize=True, channel_order="BGR"
):
    import matplotlib.pyplot as plt

    # if image is in CHW format. Change it to HWC for plotting and RBG order
    if order == "CHW":
        image = image.swapaxes(0, 1).swapaxes(1, 2)
    if channel_order == "BGR":
        image = image[:, :, [2, 1, 0]]
    if normalize:
        image = normalize_image_scale(image)
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(image, aspect="equal")
    plt.axis("off")
    plt.tight_layout()
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(
        os.path.join(path, name),
        bbox_inches="tight",
        dpi="figure",
        transparent="True",
        pad_inches=0,
    )
    logger.info("image saved: {}".format(os.path.join(path, name)))
