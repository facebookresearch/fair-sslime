#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from torchvision import transforms as transforms

TORCHVISION_TRANSFORMS = {
    "CenterCrop": transforms.CenterCrop,
    "ColorJitter": transforms.ColorJitter,
    "Grayscale": transforms.Grayscale,
    "Normalize": transforms.Normalize,
    "RandomHorizontalFlip": transforms.RandomHorizontalFlip,
    "RandomResizedCrop": transforms.RandomResizedCrop,
    "Resize": transforms.Resize,
    "ToTensor": transforms.ToTensor,
}


class TorchVisionTransformsWrapper(object):
    def __init__(self, transform, indices, *args):
        self.transform = TORCHVISION_TRANSFORMS[transform](*args)
        self.indices = set(indices)

    def __call__(self, sample):
        # Run on all indices if empty set is passed.
        indices = self.indices if self.indices else set(range(len(sample["data"])))
        for i_data in range(len(sample["data"])):
            if i_data in indices:
                sample["data"][i_data] = self.transform(sample["data"][i_data])
        return sample
