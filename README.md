# FAIR SSLIME

## Introduction
We present Self-Supervised Learning Integrated Multi-modal Environment (SSLIME), a toolkit based on PyTorch that aims to accelerate research cycle in self-supervised learning: from designing a new self-supervised task to evaluating the learned representations. The toolkit treats multiple data modalities (images, videos, audio, text) as first class citizens. The toolkit aims to provide reference implementations of several self-supervised pretext tasks and also provides an extensive benchmark suite for evaluating self-supervised representations. The toolkit is designed to be easily reusable, extensible and enable reproducible research. The toolkit also aims to support efficient distributed training across multiple nodes to facilitate research on Facebook scale data.

<p align="center">
  <img src="demo/framework_components.png" alt="Framework Components" title="Framework Components"/>
</p>

<p align="center">
  <img src="demo/framework_features.png" alt="Framework Features" title="Framework Features"/>
</p>

Currently, the toolkit supports the Rotation [1] Pretext task and evaluation of features from different layers. Support for Jigsaw, Colorization and DeepCluster pretext tasks will be added in the coming months.

## Installation

Please find installation instructions in [`INSTALL.md`](INSTALL.md).

## Getting Started

After installation, please see [`GETTING_STARTED.md`](GETTING_STARTED.md) for how to run various ssl tasks.

## License

sslime is CC-NC 4.0 International licensed, as found in the LICENSE file.

## Citation

If you use sslime in your research, please use the following BibTeX entry.

```
@article{sslime2019,
  title={SSLIME: A Toolkit for Multi-Modal Self-Supervised Training and Benchmarking},
  author={Bhooshan, Suvrat and Misra, Ishan and Fergus, Rob and Goyal, Priya},
  year={2019}
}
```

## References
- Gidaris, Spyros, Praveer Singh, and Nikos Komodakis. "Unsupervised representation learning by predicting image rotations." arXiv preprint arXiv:1803.07728 (2018).
