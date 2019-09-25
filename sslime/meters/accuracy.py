#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pprint

import torch

from sslime.utils.utils import is_pos_int


class AccuracyMeter(object):
    """Meter to calculate top-k accuracy for single label
       image classification task.
    """

    def __init__(self, topk):
        """
        args:
            topk: list of int `k` values.
        """
        assert isinstance(topk, list), "topk must be a list"
        assert len(topk) > 0, "topk list should have at least one element"
        assert [is_pos_int(x) for x in topk], "each value in topk must be >= 1"
        self._topk = topk
        self._total_correct_predictions_k = None
        self._total_sample_count = None
        # Initialize all values properly
        self.reset()

    @property
    def name(self):
        return "accuracy"

    @property
    def value(self):
        return {
            "top_{}".format(k): (correct_prediction_k / self._total_sample_count).item()
            if self._total_sample_count
            else 0.0
            for k, correct_prediction_k in zip(
                self._topk, self._total_correct_predictions_k
            )
        }

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
        if type(model_output) == list:
            assert len(model_output) == 1
            model_output = model_output[0]
        _, pred = model_output.topk(max(self._topk), dim=1, largest=True, sorted=True)

        correct_predictions = pred.eq(target.unsqueeze(1).expand_as(pred))
        for i, k in enumerate(self._topk):
            self._total_correct_predictions_k[i] += (
                correct_predictions[:, :k].float().sum().item()
            )
        self._total_sample_count += model_output.shape[0]

    def reset(self):
        self._total_correct_predictions_k = torch.zeros(len(self._topk))
        self._total_sample_count = torch.zeros(1)


class AccuracyListMeter:
    """Meter to calculate top-k accuracy for single label
       image classification task.
    """

    def __init__(self, num_list, topk):
        """
        args:
            num_list: num outputs
            topk: list of int `k` values.
        """
        assert is_pos_int(num_list), "num list must be positive"
        assert isinstance(topk, list), "topk must be a list"
        assert len(topk) > 0, "topk list should have at least one element"
        assert [is_pos_int(x) for x in topk], "each value in topk must be >= 1"
        self._num_list = num_list
        self._topk = topk
        self._meters = [AccuracyMeter(self._topk) for _ in range(self._num_list)]
        self.reset()

    @property
    def name(self):
        return "accuracylist"

    @property
    def value(self):
        val_dict = {}
        for ind, meter in enumerate(self._meters):
            meter_val = meter.value
            sample_count = meter._total_sample_count
            val_dict[ind] = {}
            val_dict[ind]["val"] = meter_val
            val_dict[ind]["sample_count"] = sample_count
        # also create dict wrt top-k
        for k in self._topk:
            top_k_str = "top_%d" % (k)
            val_dict[top_k_str] = []
            for ind in range(len(self._meters)):
                val_dict[top_k_str].append(val_dict[ind]["val"][top_k_str])
        return val_dict

    def __repr__(self):
        value = self.value
        # convert top_k list into csv format for easy copy pasting
        for k in self._topk:
            top_k_str = "top_%d" % (k)
            hr_format = ["%.1f" % (100 * x) for x in value[top_k_str]]
            value[top_k_str] = ",".join(hr_format)

        repr_dict = {"name": self.name, "num_list": self._num_list, "value": value}
        return pprint.pformat(repr_dict, indent=2)

    def update(self, model_output, target):
        """
        args:
            model_output: tensor of shape (B, C) where each value is
                          either logit or class probability.
            target:       tensor of shape (B).
            Note: For binary classification, C=2.
        """
        assert isinstance(model_output, list)
        assert len(model_output) == self._num_list
        for (meter, output) in zip(self._meters, model_output):
            meter.update(output, target)

    def reset(self):
        [x.reset() for x in self._meters]
