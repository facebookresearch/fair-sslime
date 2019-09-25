#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from sslime.meters.accuracy import AccuracyMeter, AccuracyListMeter
from sslime.meters.loss import LossMeter

METERS = {
    "accuracy": AccuracyMeter,
    "accuracy_list": AccuracyListMeter,
    "loss": LossMeter,
}
