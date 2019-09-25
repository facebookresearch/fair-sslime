#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn


class CrossEntropyMultipleOutputSingleTargetLoss(nn.Module):
    def __init__(self):
        """Intializer for the sum cross-entropy loss criterion. For a single
        tensor, this is equivalent to the cross-entropy loss. For a
        list of tensors, this computes the sum of the cross-entropy
        losses for each tensor in the list against the target.

        Config params:
        'weight': weight of sample, optional,
        'ignore_index': sample should be ignored for loss, optional,
        'reduction': specifies reduction to apply to the output, optional,
        """
        super(CrossEntropyMultipleOutputSingleTargetLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, output, target):
        assert isinstance(
            output, list
        ), "Model output should be a list of tensors. Got Type {}".format(type(output))
        assert torch.is_tensor(target), "Target should be a tensor. Got Type {}".format(
            type(target)
        )
        loss = 0
        for pred in output:
            loss += self.loss_fn(pred, target)
        return loss
