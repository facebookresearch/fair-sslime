# Using FAIR SSLIME

We provide a brief tutorial for running various evaluations using various tasks (benchmark/legacy) on various datasets.

- For installation, please refer to [`INSTALL.md`](INSTALL.md).

## Preparing Data input files

Below are the example commands to prepare input data files for various datasets.


### Preparing ImageNet-1K data files
We assume the downloaded data to look like:

```
imagenet_full_size
|_ train
|  |_ <n0......>
|  |  |_<im-1-name>.JPEG
|  |  |_...
|  |  |_<im-N-name>.JPEG
|  |_ ...
|  |_ <n1......>
|  |  |_<im-1-name>.JPEG
|  |  |_...
|  |  |_<im-M-name>.JPEG
|  |  |_...
|  |  |_...
|_ val
|  |_ <n0......>
|  |  |_<im-1-name>.JPEG
|  |  |_...
|  |  |_<im-N-name>.JPEG
|  |_ ...
|  |_ <n1......>
|  |  |_<im-1-name>.JPEG
|  |  |_...
|  |  |_<im-M-name>.JPEG
|  |  |_...
|  |  |_...
```

```bash
python extra_scripts/create_imagenet_data_files.py \
    --data_source_dir /path/to/imagenet_full_size/ \
    --output_dir /path/to/output/imagenet_handles/
```
## Running Pretext Evaluation

We provide a config to train features using the pretext rotation task on the resnet50 model. Change the output path in the config to where the imagenet handles are saved from the above script.

```bash
python tools/train.py --config_file configs/tasks/pretext_resnet_rotation_imagenet.yaml
```

## Evaluating Features

To evaluate the features trained on a model, run the following command. Change the CHECKPOINT.FEATURE_EXTRACTOR_PARAMS in the config to point to the model checkpoint you want to use.

```bash
python tools/train.py --config_file configs/tasks/eval_resnet_rotation_imagenet.yaml
```
