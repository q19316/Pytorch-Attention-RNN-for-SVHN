""" Define the network architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import cnn


class MultiLayerGRUCell(nn.Module):
    """
    Stack multiple GRU cells. For DecoderRNN.
    """
    def __init__(self, input_size, hidden_size, num_layers, drop_p):
        """
        Args:
            input_size (integer): Input size of GRU cells.
            hidden_size (integer): Hidden layer size of GRU cells.
            num_layers (integer): Number of layers to stack.
            drop_p (float): Probability to drop elements at Dropout layers.
        """
        super(MultiLayerGRUCell, self).__init__()

        self.cells = nn.ModuleList([])
        for i in range(num_layers):
            if i==0:
                self.cells.append(nn.GRUCell(input_size, hidden_size))
            else:
                self.cells.append(nn.GRUCell(hidden_size, hidden_size))
        self.dropouts = nn.ModuleList([nn.Dropout(drop_p) for _ in range(num_layers-1)])
        self.num_layers = num_layers

    def forward(self, x, h):
        """
        One step forward pass.
        
        Args:
            x (torch.FloatTensor, [batch_size, input_size]): The input features of current time step.
            h (torch.FloatTensor, [num_layers, batch_size, hidden_size]): The hidden state of previous time step.
            
        Returns:
            outputs (torch.FloatTensor, [num_layers, batch_size, hidden_size]): The hidden state of current time step.
        """
        outputs = []
        for i in range(self.num_layers):
            if i==0:
                x = self.cells[i](x, h[i])
            else:
                x = self.cells[i](self.dropouts[i-1](x), h[i])
            outputs.append(x)
        outputs = torch.stack(outputs, dim=0)
        return outputs


class DecoderRNN(nn.Module):
    """
    A decoder RNN which applies Luong attention (https://arxiv.org/abs/1508.04025).
    """
    def __init__(self, n_outputs, ftrsC, hidden_size, drop_p):
        """
        Args:
            n_outputs (integer): Softmax classes.
            ftrsC (integer): Channels of EncoderCNN's output feature map.
            hidden_size (integer): Size of GRU cells.
            drop_p (float): Probability to drop elements at Dropout layers.
        """
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.ftrsC = ftrsC
        self.embed = nn.Embedding(n_outputs, hidden_size)
        self.cell = MultiLayerGRUCell(2 * hidden_size,
                                      hidden_size,
                                      num_layers=2,
                                      drop_p=drop_p)
        self.init_state = torch.nn.Parameter(torch.randn([2, 1, hidden_size]))
        self.attn_W = nn.Linear(ftrsC, hidden_size)
        self.attn_U = nn.Linear(hidden_size, hidden_size)
        self.attn_v = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(ftrsC + hidden_size, hidden_size)
        self.drop = nn.Dropout(drop_p)
        self.classifier = nn.Linear(hidden_size, n_outputs)

    def forward(self, encoder_ftrs, sequences, test):
        """
        The forwarding behavior depends on the argument 'test'.

        Args:
            encoder_ftrs (torch.FloatTensor, [batch_size, ftrsC, ftrsH, ftrsW]): EncoderCNN's output feature map.
            sequences (torch.LongTensor, [batch_size, padded_len_tgt]): Padded target sequences. Only required when the
                                                                        argument 'test' is Fasle.
            test (bool): Determine the forwarding behavior. When test=False, it receives parallel pairs as training
                         examples and compute cross-entropy loss. But when test=True, it only needs the input image
                         and outputs the predictions along with the attention weights. 

        Returns:
            * When test=False it returns 'loss'. But when test=True, it returns a list of predicted digits and the
            attention weights.
            loss (float): The cross-entropy loss to maximizing the probability of generating ground-truth.
            predictions (list(integer)): The predictions.
            all_attn_weights (list(torch.FloatTensor)): A list contains attention alignment weights for the predictions.
        """
        batch_size = encoder_ftrs.shape[0]
        states = encoder_ftrs.view(batch_size, self.ftrsC, -1).transpose(1,2)   # [batch_size, ftrsH * ftrsW, ftrsC]
        
        initial_state = self.init_state.repeat([1, batch_size, 1])   # [2, batch_size, hidden_size]

        if test:
            assert batch_size == 1, ("Batch size should be 1 during inference.")
            all_attn_weights = []
            time_step = -1
            while (time_step < 100):
                if time_step == -1:
                    h = initial_state
                else:
                    x = torch.tensor(predictions[-1]).cuda()
                    x = self.embed(x).unsqueeze(0)                     # [1, hidden_size]
                    h = self.cell(torch.cat([y, x], dim=-1), h)        # [2, 1, hidden_size]
                attns, attn_weights = self.apply_attn(states, h[-1])   # [1, ftrsC], [1, ftrsH * ftrsW]
                y = torch.cat([attns, h[-1]], dim=-1)                  # [1, ftrsC + hidden_size]
                y = F.relu(self.fc(y))                                 # [1, hidden_size]

                if time_step > -1:
                    all_attn_weights.append(attn_weights.squeeze())
                    # Output
                    logits = self.classifier(y).squeeze()              # [n_outputs]
                    sample = torch.argmax(logits)   # Greedy Search
                    predictions.append(int(sample))
                    # Stop decoding after </s> is sampled.
                    if sample == 2:
                        break
                else:
                    predictions = [3]   # The first output is always <s>.
                time_step += 1
            return predictions, all_attn_weights
        else:
            xs = self.embed(sequences[:, :-1])   # [batch_size, padded_len_tgt, hidden_size]
            
            # Unrolling the forward pass
            outputs = []
            for time_step in range(-1, xs.shape[1]):
                if time_step == -1:
                    h = initial_state                                           # [2, batch_size, hidden_size]
                else:
                    h = self.cell(torch.cat([y, xs[:,time_step]], dim=-1), h)   # [2, batch_size, hidden_size]
                attns, _ = self.apply_attn(states, h[-1])                       # [batch_size, ftrsC]
                y = torch.cat([attns, h[-1]], dim=-1)                           # [batch_size, ftrsC + hidden_size]
                y = F.relu(self.fc(y))                                          # [batch_size, hidden_size]
                outputs.append(y)

            # Output
            outputs = torch.stack(outputs[1:], dim=1)   # [batch_size, padded_len_tgt, hidden_size]
            outputs = self.drop(outputs)
            outputs = self.classifier(outputs)          # [batch_size, padded_len_tgt, n_outputs]

            # Compute loss
            mask = sequences[:, 1:].gt(0)               # [batch_size, padded_len_tgt]
            loss = nn.CrossEntropyLoss()(outputs[mask], sequences[:, 1:][mask])
            return loss

    def apply_attn(self, source_states, target_states):
        """
        Apply attention.

        Args:
            source_states (torch.FloatTensor, [batch_size, ftrsH * ftrsW, ftrsC]): The padded encoder output states.
            target_state (torch.FloatTensor, [batch_size, hidden_size]): The decoder output state. 

        Returns:
            attns (torch.FloatTensor, [batch_size, ftrsC]): The attention result (weighted sum of Encoder output states).
            attn_weights (torch.FloatTensor, [batch_size, ftrsH * ftrsW]): The attention alignment weights.
        """
        # A two-layer network used for project every pair of [source_state, target_state].
        attns = self.attn_W(source_states) + self.attn_U(target_states).unsqueeze(1)   # [batch_size, ftrsH * ftrsW, hidden_size]
        attns = self.attn_v(F.relu(attns)).squeeze(2)                   # [batch_size, ftrsH * ftrsW]
        attns = F.softmax(attns, dim=-1)                                # [batch_size, ftrsH * ftrsW]
        attn_weights = attns.clone()
        attns = torch.sum(source_states * attns.unsqueeze(-1), dim=1)   # [batch_size, ftrsC]
        return attns, attn_weights


class AttentionRNN(nn.Module):
    """
    Attention RNN model at high-level view. It is made up of an EncoderCNN module and a DecoderRNN module.
    """
    def __init__(self, n_outputs, hidden_size, drop_p=0.):
        """
        Args:
            n_outputs (integer): Softmax classes.
            hidden_size (integer): Size of RNN cells.
            drop_p (float): Probability to drop elements at Dropout layers.
        """
        super(AttentionRNN, self).__init__()

        self.encoder = cnn.net(batch_norm=True)
        self.decoder = DecoderRNN(n_outputs, self.encoder.ftrsC, hidden_size, drop_p)

    def forward(self, xs, ys=None, test=False):
        """
        The forwarding behavior depends on the argument 'test'.

        Args:
            xs (torch.FloatTensor, [batch_size, 3, imgH, imgW]): Batched images.
            ys (torch.LongTensor, [batch_size, padded_length_of_target_sequence]): Padded target sequences. Only required
                when the argument 'test' is Fasle.
            test (bool): Determine the forwarding behavior. When test=False, it receives parallel pairs as training
                         examples and compute cross-entropy loss. But when test=True, it only needs the input image
                         and outputs the predictions along with the attention weights. 

        Returns:
            * When test=False it returns 'loss'. But when test=True, it returns a list of predicted digits and the
            attention weights.
            loss (float): The cross-entropy loss to maximizing the probability of generating ground-truth.
            predictions (list(integer)): The predictions.
            all_attn_weights (list(torch.FloatTensor)): A list contains attention alignment weights for each prediction.
        """
        return self.decoder(self.encoder(xs), ys, test)


def test():
    """
    Test the functionality of models.
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    
    import data
    loader, tokenizer = data.load(batch_size=64, augmentation=True, split='test')
    xs, ys = next(iter(loader))
    model = AttentionRNN(len(tokenizer.vocab), hidden_size=256).cuda()
    loss = model(xs.cuda(), ys.cuda())


if __name__ == '__main__':
    test()
