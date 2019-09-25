#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torchvision.transforms as transforms

from sslime.data.ssl_transforms.ssl_rotate import SSL_IMG_ROTATE
from sslime.data.ssl_transforms.basic_transforms_wrapper import (
    TORCHVISION_TRANSFORMS,
    TorchVisionTransformsWrapper,
)

TRANSFORMS = {"ssl_rotate": SSL_IMG_ROTATE}


def get_transform(transforms_list):
    compose = []
    for (key, *kwargs) in transforms_list:
        kwargs = kwargs[0] if kwargs else {}
        indices = kwargs["indices"] if "indices" in kwargs else []
        args = kwargs["args"] if "args" in kwargs else []
        if key in TRANSFORMS:
            compose.append(TRANSFORMS[key](indices, *args))
        elif key in TORCHVISION_TRANSFORMS:
            compose.append(TorchVisionTransformsWrapper(key, indices, *args))
        else:
            raise KeyError(
                "Unsupported Tranform {}. Currently supported are: {}".format(
                    key,
                    set(list(TRANSFORMS.keys()) + list(TORCHVISION_TRANSFORMS.keys())),
                )
            )

    return transforms.Compose(compose)
