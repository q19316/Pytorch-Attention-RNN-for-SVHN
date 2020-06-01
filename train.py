""" Train the model.
"""
import os
import time
import torch

"""
Configurations:
    BATCH_SIZE: Batch size.
    N_HIDDEN: Size of GRU cell in both the EncoderRNN and DecoderRNN.
    INITIAL_LR: The initial learning rate.
    EPOCHS: Total number of epochs to train.
    LR_MILESTONES: The epoch to decay the learning rate (x0.1 every milestone).
    DROPOUT: The probability to randomly drop neurons at Dropout layers.
    AUGMENTATION: Apply data augmentation or not.
    GPU_ID: Determine the GPU ID. Currently only support single GPU.
    SAVE_PATH: The path to the folder to save the loss history and checkpoints.
"""
BATCH_SIZE = 64
N_HIDDEN = 256
INITIAL_LR = 3e-4
EPOCHS = 20
LR_MILESTONES = [15]
DROPOUT = 0.
AUGMENTATION = True
GPU_ID = 0
SAVE_PATH = "logs/"


def print_lr(optimizer):
    """
    A helper funtion to print the solver's learning rate.
    """
    for param_group in optimizer.param_groups:
        print ("learning rate: %f" % param_group['lr'])


def log_history(save_path, step, loss):
    """
    A helper funtion to log the loss history.
    The history text file is saved as: {SAVE_PATH}/loss_history.csv
    """
    f = open(os.path.join(save_path ,'loss_history.csv'), 'a')
    f.write("%d, %f\n" % (step, loss))
    f.close()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    import data
    import build_model
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    assert torch.cuda.is_available()

    # Create dataset
    loader, tokenizer = data.load(split='train', batch_size=BATCH_SIZE, augmentation=AUGMENTATION)

    # Build model
    model = build_model.Seq2Seq(len(tokenizer.vocab), N_HIDDEN, drop_p=DROPOUT)
    model.train()
    model = model.cuda()

    # Training criteria
    optimizer = torch.optim.AdamW(model.parameters(), lr=INITIAL_LR)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_MILESTONES)

    # Training loop
    total_steps = 0
    for epoch in range(EPOCHS + 1):
        for step, (xs, ys) in enumerate(loader):
            total_steps = total_steps + 1
            loss = model(xs.cuda(), ys.cuda())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)   # Gradient clipping
            optimizer.step()

            if not step%10:
                print (time.strftime("%H:%M:%S", time.localtime()), end=' ')
                print ("epoch: %d, step: %d, loss: %.3f" % (epoch, step, loss))
                log_history(SAVE_PATH, total_steps, loss)
        scheduler.step()

        print_lr(optimizer)

        # Save model
        torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'checkpoint-%03d.pth' % epoch))


if __name__ == '__main__':
    main()
