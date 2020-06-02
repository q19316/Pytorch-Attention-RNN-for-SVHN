import torch
import os
import json
import random
import numpy as np
import torch.nn.utils.rnn as rnn_utils
import matplotlib.pyplot as plt
import torchvision.transforms as t
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchnlp.encoders.text import StaticTokenizerEncoder
from tqdm import tqdm


def transform(augmentation):
    if augmentation:
        return t.Compose([
            t.Resize((64, 64)),
            t.RandomCrop((54, 54)),
            t.ColorJitter(0.5, 0.5, 0.5, 0.5),
            t.ToTensor(),
            ])
    else:
        return t.Compose([
            t.Resize((64, 64)),
            t.CenterCrop((54, 54)),
            t.ToTensor(),
            ])


class SVHN(Dataset):
    """
    An abstract class representing a dataset. It stores file lists in __init__, and reads and preprocesses images
    in __getitem__.
    """
    def __init__(self, data, augmentation):
        """
        Args:
            data (list(dict)): The data of the partition. Each example is represented by {'path': image path, 'anno': annotation}.
            augmentation (bool): Whether to apply augmentation to the images. 
        """
        self.data = data
        self.transform = transform(augmentation)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            image (torch.FloatTensor, [3, 54, 54]): The image after preprocessing.
            label (string): The label sequence.
        """
        image = Image.open(self.data[idx]['path'])
        image = image.convert('RGB')
        
        # Crop to bounding box
        top, left, height, width = self.data[idx]['anno']['bbox']
        image = t.functional.crop(image, top, left, height, width)
        
        image = self.transform(image)
        
        label = self.data[idx]['anno']['label']
        return image, label

    def generateBatch(self, batch, tokenizer):
        """
        Generate a mini-batch of data. For DataLoader's 'collate_fn'.

        Args:
            batch (list(tuple)): A mini-batch of (image, label sequence). Note the label sequences are originally
                                 strings, and in this function they are encoded to torch.LongTensor.
            tokenizer (Pytorch-NLP’s StaticTokenizerEncoder): A tokenizer to encode/decode label sequences.

        Returns:
            xs (torch.FloatTensor, [batch_size, 3, 54, 54]): Batched images.
            ys (torch.LongTensor, [batch_size, (padded) n_tokens]): Batched label sequences.
        """
        xs, ys = zip(*batch)
        xs = torch.stack(xs, dim=0)
        ys = [tokenizer.encode(y) for y in ys]
        ys = rnn_utils.pad_sequence(ys, batch_first=True)   # [batch_size, (padded) n_tokens]
        return xs, ys


def preprocess_label(data):
    """
    For convenience, the label is transformed from a sequence of integers to a string.

    Args:
        data (list(dict)): The data of the partition. Each example is represented by {'path': image path, 'anno': annotation}.
    """
    for x in data:
        label = []
        for d in x['anno']['label']:
            if d == 10:
                label.append('0')
            else:
                label.append(str(int(d)))
        # We append <s> to the beginning of the sequence to leverage RNN training.
        label = ['<s>'] + label
        label = ' '.join(label)
        x['anno']['label'] = label
    return data


def compute_bbox(data):
    """
    Compute bounding box covering all of the digits in the image. Each is represented by [top, left, height, width].
    
    Args:
        data (list(dict)): The data of the partition. Each example is represented by {'path': image path, 'anno': annotation}.
    """
    for x in tqdm(data):
        xmin = np.min(x['anno']['left'])
        xmax = np.max(np.array(x['anno']['left']) + np.array(x['anno']['width']))
        ymin = np.min(x['anno']['top'])
        ymax = np.max(np.array(x['anno']['top']) + np.array(x['anno']['height']))

        wh = np.array([(xmax-xmin), (ymax-ymin)])
        # expand by 30%
        wh = wh * 1.3
        
        center = np.array([(xmax+xmin)/2, (ymax+ymin)/2])
        top = int(center[1] - wh[1]/2)
        left = int(center[0] - wh[0]/2)
        height = int(wh[1])
        width = int(wh[0])
        
        x['anno']['bbox'] = [top, left, height, width]
    return data

    
def load_json(split):
    """
    Args:
        split (sting): One of 'train', 'extra', or 'test'.

    Returns:
        data (list(dict)): The data of the partition. Each example is represented by {'path': image path, 'anno': annotation}.
    """
    with open('%s.json'%split, 'r') as f:
        data = json.load(f)
    random.seed(12345)
    random.shuffle(data)
    return data
    

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
    train_dataset = preprocess_label(train_dataset)

    tokenizer = StaticTokenizerEncoder([x['anno']['label'] for x in train_dataset],
                                       tokenize=lambda s: s.split(),
                                       append_eos=True,
                                       reserved_tokens=['<pad>', '<unk>', '</s>'])
    print (tokenizer.vocab)

    if split == 'train':
        dataset = train_dataset
    else:
        dataset = load_json('test')
        dataset = preprocess_label(dataset)

    print ("Compute bounding boxes ...")
    dataset = compute_bbox(dataset)
    
    dataset = SVHN(dataset, augmentation=(augmentation and split=='train'))
    print ("Dataset size:", len(dataset))
    loader = DataLoader(dataset,
                        batch_size,
                        shuffle=shuffle,
                        collate_fn=lambda batch: dataset.generateBatch(batch, tokenizer),
                        num_workers=4,
                        pin_memory=True)
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

