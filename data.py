import torch
import os
import json
import random
import torch.nn.utils.rnn as rnn_utils
import matplotlib.pyplot as plt
import torchvision.transforms as t
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchnlp.encoders.text import StaticTokenizerEncoder


def transform(augmentation):
    if augmentation:
        return t.Compose([
            t.ColorJitter(0.5, 0.5, 0.5, 0.5),
            t.RandomResizedCrop(size=(32, 64), scale=(0.7, 1.0), ratio=(1.5, 4.0)),
            t.ToTensor(),
            ])
    else:
        return t.Compose([
            t.Resize((40, 70)),
            t.CenterCrop((32, 64)),
            t.ToTensor(),
            ])


class SVHN(Dataset):
    """
    An abstract class representing a dataset. It stores file lists in __init__, and reads and preprocesses images
    in __getitem__.
    """
    def __init__(self, pairs, augmentation):
        """
        Args:
            pairs (list(dict)): Pairs of examples represented by {'path': image path, 'label': label sequence}.
            augmentation (bool): Whether to apply augmentation to the images. 
        """
        self.pairs = pairs
        self.transform = transform(augmentation)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Returns:
            x (torch.FloatTensor, [3, 32, 64]): The image after preprocessing.
            y (string): The label sequence.
        """
        x = Image.open(self.pairs[idx]['path'])
        x = x.convert('RGB')
        x = self.transform(x)
        y = self.pairs[idx]['label']
        return x, y

    def generateBatch(self, batch, tokenizer):
        """
        Generate a mini-batch of data. For DataLoader's 'collate_fn'.

        Args:
            batch (list(tuple)): A mini-batch of (image, label sequence). Note the label sequences are originally
                                 strings, and in this function they are encoded to torch.LongTensor.
            tokenizer (Pytorch-NLP’s StaticTokenizerEncoder): A tokenizer to encode/decode label sequences.

        Returns:
            xs (torch.FloatTensor, [batch_size, 3, 32, 64]): Batched images.
            ys (torch.LongTensor, [batch_size, (padded) n_tokens]): Batched label sequences.
        """
        xs, ys = zip(*batch)
        xs = torch.stack(xs, dim=0)
        ys = [tokenizer.encode(y) for y in ys]
        ys = rnn_utils.pad_sequence(ys, batch_first=True)   # [batch_size, (padded) n_tokens]
        return xs, ys


def preprocess_target(label):
    """
    For convenience, convert the sequence of integers to a string.

    Args:
        label (list(integer)): Label sequence for digits in the image.

    Returns:
        label_out (string): Label sequence after preprocessing.
    """
    label_out = []
    for l in label:
        if l == 10:
            label_out.append('0')
        else:
            label_out.append(str(int(l)))
    # We append <s> to the beginning of the sequence to leverage RNN training.
    label_out = ['<s>'] + label_out
    label_out = ' '.join(label_out)
    return label_out


def load_json(split):
    """
    Args:
        split (sting): One of 'train', 'extra', or 'test'.

    Returns:
        pairs (list(dict)): The data of the partition. Each example is represented by {'path': image path, 'label': label sequence}.
    """
    with open('%s.json'%split, 'r') as f:
        data = json.load(f)
    pairs = [{'path': os.path.join(split, x[0]), 'label': preprocess_target(x[1]['label'])} for x in data]
    random.seed(4321)
    random.shuffle(pairs)
    return pairs


def load(batch_size, augmentation, split, shuffle=True):
    """
    Args:
        split (string): Which of the subset of data to take. One of 'train' or 'test'.
        batch_size (integer): Batch size.
        augmentation (bool): Whether to apply data augmentation. Only work on training set.

    Return:
        loader (DataLoader): A DataLoader can generate batches of (image, label sequence).
        tokenizer (Pytorch-NLP’s StaticTokenizerEncoder): A tokenizer to encode/decode label sequences.
    """
    assert split in ['train', 'test']

    train_dataset = load_json('train') + load_json('extra')
    train_labels = [pair['label'] for pair in train_dataset]

    tokenizer = StaticTokenizerEncoder(train_labels,
                                       tokenize=lambda s: s.split(),
                                       append_eos=True,
                                       reserved_tokens=['<pad>', '<unk>', '</s>'])
    print (tokenizer.vocab)

    if split == 'train':
        dataset = train_dataset
    else:
        dataset = load_json('test')
    dataset = SVHN(dataset, augmentation=(augmentation and split=='train'))
    print ("Dataset size:", len(dataset))

    loader = DataLoader(dataset,
                        batch_size,
                        shuffle=shuffle,
                        collate_fn=lambda batch: dataset.generateBatch(batch, tokenizer),
                        num_workers=4)
    return loader, tokenizer


def inspect_data():
    loader_raw, _ = load(batch_size=64, augmentation=False, split='train', shuffle=False)
    loader, tokenizer = load(batch_size=64, augmentation=True, split='train', shuffle=False)
    xs_raw, _ = next(iter(loader_raw))
    xs, ys = next(iter(loader))
    print (xs.shape, ys.shape)
    for i in range(64):
        print (tokenizer.decode(ys[i]))
        plt.figure()
        plt.subplot(2,1,1)
        plt.imshow(xs_raw[i].permute(1,2,0))
        plt.subplot(2,1,2)
        plt.imshow(xs[i].permute(1,2,0))
        plt.show()


if __name__ == '__main__':
    inspect_data()

