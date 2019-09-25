#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

from sslime.criterion import get_criterion


class LossMeter(object):
    """Meter to calculate average UserDefined Loss
    """

    def __init__(self):
        """
        args:
            topk: list of int `k` values.
        """
        self._criterion = get_criterion()
        self._losses = None
        # Initialize all values properly
        self.reset()

    @property
    def name(self):
        return "loss"

    @property
    def value(self):
        return np.mean(self._losses)

    def __repr__(self):
        return repr({"name": self.name, "value": self.value})

    def update(self, model_output, target):
        """
        args:
            model_output: tensor of shape (B, C) where each value is
                          either logit or class probability.
            target:       tensor of shape (B).
            Note: For binary classification, C=2.
        """
        loss = self._criterion(model_output, target)
        self._losses.append(loss.item())

    def reset(self):
        self._losses = []
