#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torchvision.transforms.functional as TF


class SSL_IMG_ROTATE(object):
    def __init__(self, indices, num_angles=4, num_rotations_per_img=1):
        self.indices = set(indices)
        self.num_angles = num_angles
        self.num_rotations_per_img = num_rotations_per_img
        # the last angle is 360 and 1st angle is 0, both give the original image.
        # 360 is not useful so remove it
        self.angles = torch.linspace(0, 360, num_angles + 1)[:-1]

    def __call__(self, sample):
        data, labels = [], []
        indices = self.indices if self.indices else set(range(len(sample["data"])))
        for idx in range(len(sample["data"])):
            if idx in indices:
                for _ in range(self.num_rotations_per_img):
                    label = torch.randint(self.num_angles, [1]).item()
                    img = TF.rotate(sample["data"][idx], self.angles[label])
                    data.append(img)
                    labels.append(label)

        sample["data"] = data
        sample["label"] = labels

        return sample
