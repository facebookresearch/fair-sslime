#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from sslime.models.heads.evaluation_mlp import Eval_MLP
from sslime.models.heads.mlp import MLP

HEADS = {"eval_mlp": Eval_MLP, "mlp": MLP}
