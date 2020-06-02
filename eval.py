""" Compute the loss and phoneme error rate (PER).
"""
import torch
import os


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


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    import data
    import build_model
    assert torch.cuda.is_available()

    # Create dataset
    loader, tokenizer = data.load(split=SPLIT, batch_size=1, augmentation=False)


    # Build model
    model = build_model.Seq2Seq(len(tokenizer.vocab), N_HIDDEN)
    model.load_state_dict(torch.load(CKPT_FILE))
    model.eval()
    model = model.cuda()

    # Inference
    total_loss = 0
    n_tokens = 0
    correct = 0
    with torch.no_grad():
        for step, (x, y) in enumerate(loader):
            total_loss += model(x.cuda(), y.cuda()) * y.shape[1]
            n_tokens += y.shape[1]

            predictions, _ = model(x.cuda(), test=True)
            predictions = tokenizer.decode(predictions)
            ground_truth = tokenizer.decode(y[0])
            correct += int(predictions == ground_truth)
            print ("step: %d, loss: %.3f, accuracy: %.4f"
                   % (step+1, total_loss/n_tokens, correct/(step+1)), end='\r')
        print ()


if __name__ == '__main__':
    main()
