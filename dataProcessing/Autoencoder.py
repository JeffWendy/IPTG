import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import time
import random
from utils import timeSince, showPlot, read_trajs
import pickle as pkl
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, SOT):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.SOT = SOT

    def forward(self, encoder_outputs, encoder_hidden, length_limit, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(self.SOT)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(length_limit):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: Use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden

class DataLoader():
    def __init__(self, data_path, batch_size, SOT, EOT, shuffle=False):
        self.dataset = []
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.EOT = EOT
        self.SOT = SOT
        self.idx = 0

        trajs, max_len = read_trajs(data_path)
        
        tmp_batch = []
        for l in range(1, max_len + 1):
            if not (l in trajs):
                continue
            for traj in trajs[l]:
                if len(tmp_batch) >= batch_size:
                    self.dataset.append(self._pad_sequence(tmp_batch).to(device))
                    tmp_batch = []
                tmp_batch.append(torch.LongTensor(traj))

        if len(tmp_batch) > 0:
            self.dataset.append(self._pad_sequence(tmp_batch).to(device))

        self.batch_count = len(self.dataset)

    def _shuffle(self):
        random.shuffle(self.dataset)

    def _pad_sequence(self, seq_list, batch_first=True):
        tmp_batch = pad_sequence(seq_list, batch_first=batch_first, padding_value=self.EOT)
        prefix = torch.ones((len(seq_list), 1), dtype=torch.int64) * self.SOT
        suffix = torch.ones((len(seq_list), 2), dtype=torch.int64) * self.EOT
        return torch.cat((prefix, tmp_batch, suffix), dim=1)
    
    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.batch_count - 1:
            if self.shuffle:
                self._shuffle()
            raise StopIteration()

        self.idx += 1
        return (self.dataset[self.idx], self.dataset[self.idx])
    
class Window():
    def __init__(self, size):
        self.size = size
        self.window = []
        
    def mean(self):
        if len(self.window) == 0:
            return None
        return sum(self.window) / len(self.window)
    
    def append(self, value):
        if len(self.window) >= self.size:
            self.window.pop(0)
        self.window.append(value)

def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_rate):

    total_loss = 0

    for data in dataloader:
        input_tensor, target_tensor = data
        length_limit = target_tensor.size(dim=1)
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)

        target_tensor_ = None
        if random.random() > 1 - teacher_forcing_rate:
            target_tensor_ = target_tensor

        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, length_limit, target_tensor_)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / dataloader.batch_count

def train(train_dataloader, encoder, decoder, n_epochs, 
          learning_rate=0.001, print_every=10, plot_every=10, 
          teacher_forcing_rate=1, early_stop_interval=30, window_size=5):
    start = time.time()
    plot_losses = []
    plot_x_axis = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    epoch_teacher_forcing_rate = teacher_forcing_rate
    last_loss = 100000
    cur_epoch = 0

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    window = Window(window_size)
    try:
        for epoch in range(1, n_epochs + 1):
            cur_epoch = epoch
            epoch_teacher_forcing_rate = max(teacher_forcing_rate * (0.8*n_epochs - epoch) / (0.8*n_epochs), 0)
            loss = train_epoch(train_dataloader, encoder, decoder, 
                            encoder_optimizer, decoder_optimizer, criterion, epoch_teacher_forcing_rate)
            window.append(loss)
            print_loss_total += loss
            plot_loss_total += loss

            if epoch % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                            epoch, epoch / n_epochs * 100, print_loss_avg))

            if epoch % plot_every == 0:
                plot_x_axis.append(epoch)
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
            
            if epoch > 0 and epoch % 100 == 0:
                torch.save(encoder, 'encoder_{}.torch'.format(cur_epoch))
                torch.save(decoder, 'decoder_{}.torch'.format(cur_epoch))                

            """
            if epoch > 0 and epoch % early_stop_interval == 0:
                mean = window.mean()
                if last_loss < mean - 0.001:
                    break
                last_loss = mean
            """

    except Exception as e:
        print(e)
    finally:
        torch.save(encoder, 'encoder_{}.torch'.format(cur_epoch))
        torch.save(decoder, 'decoder_{}.torch'.format(cur_epoch))

def encode_trajs(data_path, encoder, embedding_path, SOT, EOT):
    trajs, max_len = read_trajs(data_path)
    embedding = []
    for l in range(1, max_len + 1):
        if not (l in trajs):
            continue
        tmp_trajs = torch.LongTensor(trajs[l])
        prefix = torch.ones((tmp_trajs.size(0), 1), dtype=torch.int64) * SOT
        suffix = torch.ones((tmp_trajs.size(0), 1), dtype=torch.int64) * EOT
        tmp_trajs = torch.cat((prefix, tmp_trajs, suffix), dim=1).to(device)
        _, hidden = encoder(tmp_trajs)
        embedding.append(torch.squeeze(hidden, 0))
    
    embedding = torch.cat(embedding, dim=0).detach().cpu().numpy()

    with open(embedding_path, 'wb') as f:
        pkl.dump(embedding, f)


if __name__ == '__main__':
    # 固定参数
    data_path = 'real.data'
    SOT = 2751
    EOT = 2752
    batch_size = 32
    input_size = output_size = 2753
    dataloader = DataLoader(data_path, batch_size, SOT, EOT, shuffle=True)

    # 超参数
    dropout = 0.1
    teacher_forcing_rate = 1
    hidden_size = 128
    n_epochs = 1000
    learning_rate = 0.001

    # 模型
    encoder = EncoderRNN(input_size, hidden_size, dropout_p=0.1).to(device)
    decoder = DecoderRNN(hidden_size, output_size, SOT).to(device)
    train(dataloader, encoder, decoder, n_epochs, learning_rate, teacher_forcing_rate=teacher_forcing_rate, early_stop_interval=30, 
          print_every=10, plot_every=10, window_size=5)
    encode_trajs(data_path, encoder, './data/traj_embedding_numpy_array.pkl', SOT, EOT)