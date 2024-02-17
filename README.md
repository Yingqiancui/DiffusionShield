# DiffusionShield

DiffusionShield offers a novel approach to copyright protection for digital images against generative diffusion models. This tool allows users to embed watermarks into images and detect them, providing an added layer of security and authenticity.

## Paper Reference

For detailed insights and the methodology behind DiffusionShield, refer to the paper: [DiffusionShield: A Watermark for Copyright Protection against Generative Diffusion Models](https://arxiv.org/abs/2306.04642).

## Getting Started

Before you can start using DiffusionShield to train watermark patches, add watermarks to images, or detect watermarks, ensure you have the necessary environment and dependencies set up.

### Training Watermark Patches and Detector

To train the Watermark Patches and Detector, execute the following command:

```sh
sh train.sh
```
### Adding Watermark to Images

To embed a watermark into your images, use the command:
```sh
python add_watermark.py
```
### Detecting Watermarks on Images

For detecting watermarks on images and calculating the bit accuracy, run:
```sh
python test_acc.py
```
