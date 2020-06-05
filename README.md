# Attention-RNN-for-SVHN

This code implements attention-based RNN to recognize multi-digit numbers from SVHN dataset.

Similar to [Goodfellow _et al._](http://arxiv.org/pdf/1312.6082.pdf), our model runs directly on the entire sequence without resorting to character segmentation.
The idea is inspired by [Xu _et al._](https://arxiv.org/pdf/1502.03044.pdf), which proposed a model to **automatically learns where to look** when generating corresponding text for an image.
The entire system has two compoents: A **CNN encoder** that extracts visual features from the input images, and a **attention-based RNN decoder** that emits digit sequence as outputs.

With our default recipe, it would only take ~2 hours to complete the training.
After that you should achieve **~96.2%** test accuracy, which is slightly better than [Goodfellow et al.](http://arxiv.org/pdf/1312.6082.pdf) (96.03%).

You can also generalize this method to other OCR tasks such as license plate recognition or text transcription.

## Network Architecture



## Requirements

* Python 3.6
* Pytorch 1.4
* torchvision 0.6.0
* [PyTorch-NLP](https://github.com/PetrochukM/PyTorch-NLP) 0.5.0
* h5py 2.10.0
* matplotlib
* Pillow
* tqdm

## Usage
### Data
Download SVHN dataset format 1 (train.tar.gz, test.tar.gz , extra.tar.gz) from http://ufldl.stanford.edu/housenumbers/ .
We would use `train` set and `extra` set for training and evaluate the final performance on `test` set.

### Convert digitStruct.mat to json.
After extracting the *.tar.gz files, you can find `digitStruct.mat` in each folder.
The digitStruct.mat contains all of the annotation infomations for the images.
However accessing *.mat files can be slow and inefficient, so we would like convert the digitStruct.mat files to individual json files.
Modify the `ROOT` in `prepare_data.py` to your path containing the folders (`train/`, `extra/`, `test/`), and run the code:

```
python prepare_data.py
```
