#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import os
from PIL import Image

from torch.utils.data import Dataset

from sslime.core.config import config as cfg
from sslime.core.config import logger
from sslime.utils.utils import get_mean_image


class DiskImageDataset(Dataset):
    """Base Dataset class for loading images from Disk."""

    def __init__(self, path, split):
        assert os.path.exists(path)

        self.split = split
        if cfg[split].MMAP_MODE:
            self.paths = np.load(path, mmap_mode="r")
        else:
            self.paths = np.load(path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Returns (image, is_success)
        is_success = True
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception as e:
            if cfg.VERBOSE:
                logger.warning(e)
            img = get_mean_image(cfg[self.split].DEFAULT_GRAY_IMG_SIZE)
            is_success = False

        return img, is_success

    def num_samples(self):
        return len(self.paths)
