#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
This file specifies the config options for SSL framework. You should write a yaml
file which specifies the values for parameters in this file. Do NOT change the
default values in this file.
"""
import copy
import logging
import numpy as np
import pprint
import six
import yaml
from ast import literal_eval

from sslime.utils.collections import AttrDict

# create the logger
logger = logging.getLogger(__name__)

__C = AttrDict()
config = __C

# Training Data Options
__C.TRAIN = AttrDict()
# Sources for reading data.
# Currently supports: disk
# Parallel aligned with DATA_PATHS argument.
__C.TRAIN.DATA_SOURCES = ["disk"]
__C.TRAIN.DATA_PATHS = ["/path/to/data.npy"]
__C.TRAIN.LABEL_SOURCES = ["disk"]
__C.TRAIN.LABEL_PATHS = ["/path/to/labels.npy"]
__C.TRAIN.MMAP_MODE = True
__C.TRAIN.EVERSTORE_RETRY = 3
__C.TRAIN.DEFAULT_GRAY_IMG_SIZE = 224
__C.TRAIN.BATCH_SIZE = 256
__C.TRAIN.TRANSFORMS = []

__C.TEST = AttrDict()
__C.TEST.DATA_SOURCES = ["disk"]
__C.TEST.DATA_PATHS = ["/path/to/val_images.npy"]
__C.TEST.LABEL_SOURCES = ["disk"]
__C.TEST.LABEL_PATHS = ["/path/to/val_labels.npy"]
__C.TEST.MMAP_MODE = True
__C.TEST.EVERSTORE_RETRY = 3
__C.TEST.DEFAULT_GRAY_IMG_SIZE = 224
__C.TEST.BATCH_SIZE = 256
__C.TEST.TRANSFORMS = []

# Model Parameters
__C.MODEL = AttrDict()
__C.MODEL.TRUNK = AttrDict()
__C.MODEL.TRUNK.TYPE = "alexnet"
__C.MODEL.TRUNK.LINEAR_FEAT_POOL_OPS = [
    ("AdaptiveMaxPool2d", [12]),
    ("AdaptiveMaxPool2d", [7]),
    ("AdaptiveMaxPool2d", [5]),
    ("AdaptiveMaxPool2d", [6]),
    ("AdaptiveMaxPool2d", [6]),
]
__C.MODEL.TRUNK.SHOULD_FLATTEN = False

__C.MODEL.HEAD = AttrDict()
# List of Pairs:
# Pair[0] = Type of Head.
# Pair[1] = kwargs passed to head constructor.
__C.MODEL.HEAD.PARAMS = []
__C.MODEL.HEAD.APPLY_BATCHNORM = True
__C.MODEL.HEAD.BATCHNORM_EPS = 1e-5
__C.MODEL.HEAD.BATCHNORM_MOMENTUM = 0.1

__C.MODEL.FEATURE_EVAL_MODE = False
__C.MODEL.EVAL_FEATURES = []

# Training Parameters
__C.TRAINER = AttrDict()
__C.TRAINER.MAX_EPOCHS = 100
# After every how many epochs to run validation loop
__C.TRAINER.EVAL_PERIOD = 1

__C.TRAINER.CRITERION = "cross_entropy_multiple_output_single_target"
__C.TRAINER.TRAIN_LOOP = "generic_train_loop"
__C.TRAINER.TRAIN_METERS = {"accuracy_list": {"num_list": 1, "topk": [1]}}

__C.TRAINER.EVAL_MODEL = True
__C.TRAINER.EVAL_LOOP = "generic_eval_loop"
__C.TRAINER.EVAL_METERS = {"accuracy_list": {"num_list": 1, "topk": [1]}}
__C.TRAINER.NUM_WORKERS = 40
__C.TRAINER.LOG_ITER_PERIOD = 100
__C.TRAINER.AUTO_RESUME = True
__C.TRAINER.RESET_START_EPOCH = False

__C.GPU_IDS = []

__C.OPTIMIZER = AttrDict()
__C.OPTIMIZER.TYPE = "sgd"
__C.OPTIMIZER.BASE_LR = 0.1
__C.OPTIMIZER.MOMENTUM = 0.0
__C.OPTIMIZER.WEIGHT_DECAY = 0.0
__C.OPTIMIZER.DAMPENING = 0
__C.OPTIMIZER.NESTEROV = False
__C.OPTIMIZER.REGULARIZE_BIAS = True
__C.OPTIMIZER.REGULARIZE_BATCHNORM = False

__C.SCHEDULER = AttrDict()
__C.SCHEDULER.TYPE = "step"
__C.SCHEDULER.STEP_SIZE = 15
__C.SCHEDULER.GAMMA = 0.1

__C.CHECKPOINT = AttrDict()
__C.CHECKPOINT.CHECKPOINT_PERIOD = 1
__C.CHECKPOINT.DIR = "/path/to/save/checkpoint/location"
__C.CHECKPOINT.FEATURE_EXTRACTOR_PARAMS = ""

__C.VERBOSE = True


def merge_dicts(dict_a, dict_b, stack=None):
    for key, value_ in dict_a.items():
        full_key = ".".join(stack) + "." + key if stack is not None else key
        if key not in dict_b:
            raise KeyError("Invalid key in config file: {}".format(key))
        value = copy.deepcopy(value_)
        value = _decode_cfg_value(value)
        value = _check_and_coerce_cfg_value_type(value, dict_b[key], key, full_key)

        # recursively merge dicts
        if isinstance(value, AttrDict):
            try:
                stack_push = [key] if stack is None else stack + [key]
                merge_dicts(dict_a[key], dict_b[key], stack_push)
            except BaseException:
                logger.critical("Error under config key: {}".format(key))
                raise
        else:
            dict_b[key] = value


def cfg_from_file(filename):
    """
    Load a config file and merge it into the default options.
    """
    with open(filename, "r") as fopen:
        yaml_config = AttrDict(yaml.load(fopen, Loader=yaml.FullLoader))
    merge_dicts(yaml_config, __C)


def print_cfg():
    logger.info("Training with config:")
    logger.info(pprint.pformat(__C))


def cfg_from_list(args_list):
    """
    Set config keys via list (e.g., from command line).
    Example: `args_list = ['NUM_DEVICES', 8]`.
    """
    assert (
        len(args_list) % 2 == 0
    ), "Looks like you forgot to specify values or keys for some args"
    for key, value in zip(args_list[0::2], args_list[1::2]):
        key_list = key.split(".")
        cfg = __C
        for subkey in key_list[:-1]:
            assert subkey in cfg, "Config key {} not found".format(subkey)
            cfg = cfg[subkey]
        subkey = key_list[-1]
        assert subkey in cfg, "Config key {} not found".format(subkey)
        val = _decode_cfg_value(value)
        val = _check_and_coerce_cfg_value_type(val, cfg[subkey], subkey, key)
        cfg[subkey] = val


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, six.string_types):
        return v
    # Try to interpret `v` as: string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """
    Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b, type_a = type(value_b), type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list, dict<->attrdict
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, six.string_types):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    elif isinstance(value_a, AttrDict) and isinstance(value_b, dict):
        value_a = dict(value_a)
    else:
        raise ValueError(
            "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
            "key: {}".format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a
