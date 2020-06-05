# Attention-RNN-for-SVHN

This code implements attention-based RNN to recognize multi-digit numbers from SVHN dataset.

Similar to [Goodfellow et al.](http://arxiv.org/pdf/1312.6082.pdf), our model runs directly on the entire sequence without resorting to character segmentation.
The idea is inspired by [Xu et al.](https://arxiv.org/pdf/1502.03044.pdf), which proposed a model to **automatically learns where to look** when generating corresponding text for an image.
The entire system has two compoents: A **CNN encoder** that extracts visual features from the input images, and a **attention-based RNN decoder** that emits digit sequence as outputs.

With this training recipe, you should achieve about 96.2% test accuracy, which is slightly better than [Goodfellow et al.](http://arxiv.org/pdf/1312.6082.pdf) (96.03%).

You can also generalize this method to other OCR tasks such as license plate recognition or text transcription.