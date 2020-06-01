""" Inference on arbitrary images and visualize the attention matrix.
"""
import torch
import data
import build_model
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as t
import cv2

"""
Configurations:
    SPLIT: Specify which subset of the data to evaluate.
    N_HIDDEN: Size of GRU cell in both the EncoderRNN and DecoderRNN.
    GPU_ID: Determine the GPU ID. Currently only support single GPU.
    CKPT_FILE: The checkpoint to restore.
"""
SPLIT = 'test'
N_HIDDEN = 256
GPU_ID = 0
CKPT_FILE = "logs/checkpoint-020.pth"


def showAttention(image, attentions):
    plt.figure(figsize=(10,5))
    attentions = attentions[:-1]
    plt.subplot(1, len(attentions)+1, 1)
    plt.imshow(image)
    for i, attn in enumerate(attentions):
        attn = np.reshape(attn, [4, 8])
        attn = cv2.resize(attn, (256, 128))
        attn = np.expand_dims(attn, axis=-1)
        attn = attn * (1 / np.max(attn))
        plt.subplot(1, len(attentions)+1, i+2)
        plt.imshow((image/255.0) * attn)
        plt.axis('off')
    plt.show()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

    assert torch.cuda.is_available()

    # Create dataset
    loader, tokenizer = data.load(split=SPLIT, batch_size=1, augmentation=False)

    # Build model
    model = build_model.Seq2Seq(len(tokenizer.vocab), N_HIDDEN)
    model.load_state_dict(torch.load(CKPT_FILE))
    model.eval()
    model = model.cuda()

    # Inference
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            predictions, attentions = model(x.cuda(), test=True)
            predictions = tokenizer.decode(predictions)
            ground_truth = tokenizer.decode(y[0])
            print ("# %d" % i)
            print ("Predict:")
            print (predictions)
            print ("Ground-truth:")
            print (ground_truth)
            print ()

            if predictions != ground_truth:
                attentions = torch.stack(attentions, dim=0).cpu().numpy()   # (target_length, source_length)
                x = t.ToPILImage()(x[0])
                x = np.array(x)
                showAttention(x, attentions)


if __name__ == '__main__':
    main()
