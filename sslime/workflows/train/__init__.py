#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from sslime.workflows.train.generic_train_loop import generic_train_loop

TRAIN_LOOPS = {"generic_train_loop": generic_train_loop}
