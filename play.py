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
CKPT_FILE = "logs/checkpoint-015.pth"


def showAttention(image, attentions, predictions, ground_truth):
    plt.figure(figsize=(5,3))
    plt.subplot(1, len(attentions)+1, 1)
    plt.title("Ground-truth:\n[%s]"%ground_truth, fontsize=10)
    plt.imshow(image)
    plt.axis('off')
    for i, attn in enumerate(attentions):
        attn = np.reshape(attn, [7, 7])
        attn = cv2.resize(attn, (image.shape[1], image.shape[0]))
        attn = np.expand_dims(attn, axis=-1)
        attn = attn * (1.0 / np.max(attn))    # Scaling to (0, 1.0) for better visualization
        plt.subplot(1, len(attentions)+1, i+2)
        plt.title("Predict: %s" % predictions.split()[i], fontsize=10)
        plt.imshow((image/255.0) * attn)
        plt.axis('off')
    plt.show()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

    assert torch.cuda.is_available()

    # Create dataset
    loader, tokenizer = data.load(split=SPLIT, batch_size=1, augmentation=False)

    # Build model
    model = build_model.AttentionRNN(len(tokenizer.vocab), N_HIDDEN)
    model.load_state_dict(torch.load(CKPT_FILE))
    model.eval()
    model = model.cuda()

    # Inference
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            predictions, attentions = model(x.cuda(), test=True)
            
            predictions = tokenizer.decode(predictions[1:-1])
            ground_truth = tokenizer.decode(y[0][1:-1])
            
            attentions = torch.stack(attentions, dim=0).cpu().numpy()[:-1]   # (target_length, source_length)
            
            x = t.ToPILImage()(x[0])
            x = np.array(x)
            showAttention(x, attentions, predictions, ground_truth)


if __name__ == '__main__':
    main()
