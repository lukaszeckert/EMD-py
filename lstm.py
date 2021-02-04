import torch
from torchtext import data

torch.manual_seed(13)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from utils import plot_confusion_matrx, clean_text
import torch.nn as nn
import torch.nn.functional as F
import sys
import torch.optim as optim
import numpy as np



class Lstm(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dimension = 50
        self.rnn = nn.LSTM(embedding_dim,
                           128,
                           num_layers=2,
                           bidirectional=True,
                           dropout=0.4)

        self.fc = nn.Linear(128*2, output_dim)
        self.softmax = nn.LogSoftmax()
        self.dropout = nn.Dropout(0.4)

    def last_timestep(self, unpacked, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(unpacked.size(0),
                                               unpacked.size(2)).unsqueeze(1)
        return unpacked.gather(1, idx).squeeze()


    def forward(self, review, lengths):
        embedded = self.dropout(self.embedding(review))

        packed_input = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # out_forward = output[range(len(output)), lengths.cpu() - 1, :self.dimension]
        # out_reverse = output[:, 0, self.dimension:]
        # out_reduced = torch.cat((out_forward, out_reverse), 1)
        output = self.last_timestep(output, lengths)
        text_fea = self.dropout(output)


        text_fea = self.fc(text_fea)
       # text_fea = torch.squeeze(text_fea, 1)

        return self.softmax(text_fea)





def train(model, iterator, optimizer, loss_function):
    losses = []
    acc = []
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(*batch.reviewText)
        loss = loss_function(predictions, batch.score.flatten())

        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
        acc.extend((predictions.argmax(axis=-1) == batch.score).detach().cpu().numpy())
    return np.mean(losses), np.mean(acc)

def evaluate(model, iterator, loss_function):
    losses = []
    acc = []
    l1_loss = []
    model.eval()
    for batch in iterator:
        predictions = model(*batch.reviewText)
        loss = loss_function(predictions, batch.score.flatten())
        losses.append(loss.detach().cpu().numpy())
        acc.extend((predictions.argmax(axis=-1) == batch.score).detach().cpu().numpy())
        l1_loss.extend(torch.abs(predictions.argmax(axis=-1) - batch.score).detach().cpu().numpy())
    return np.mean(losses), np.mean(acc), np.mean(l1_loss)


def build_model(load=False):
    text = data.Field(sequential=True, lower=True, preprocessing=lambda x: clean_text(x), include_lengths=True,batch_first=True)
    label = data.Field(use_vocab = False, sequential=False, preprocessing=lambda x: int(float(x)))

    fields = [
        ("reviewerID", None),
        ("asin", None),
        ("reviewerName", None),
        ("helpful", None),
        ("reviewText", text),
      ('summary', None),
        ("helpful", None),
        ("helpful", None),
      ('score', label)
    ]

    # load the dataset in json format
    train_ds, = data.TabularDataset.splits(
       path = 'data',
       train = 'train_under.csv',
       format = 'csv',
       fields = fields,
       skip_header = True
        )
    # valid_ds, =  data.TabularDataset.splits(
    #    path = 'data',
    #    validation = 'eval.csv',
    #    format = 'csv',
    #    fields = fields,
    #    skip_header = True
    #     )

    text.build_vocab(train_ds, vectors="glove.6B.100d", unk_init = torch.Tensor.normal_)
    label.build_vocab(train_ds)

    print(len(text.vocab))

    # train_iterator, = data.BucketIterator.splits(
    #     (train_ds,),
    #     batch_size=256,
    #     device=device, shuffle=True)

    # valid_iterator, = data.BucketIterator.splits((valid_ds,), batch_size=256, device=device, shuffle=False)

    INPUT_DIM = len(text.vocab)
    EMBEDDING_DIM = 100
    OUTPUT_DIM = 6
    PAD_IDX = text.vocab.stoi[text.pad_token]

    model = Lstm(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)

    model.embedding.weight = nn.Parameter(text.vocab.vectors, requires_grad=False)
    UNK_IDX = text.vocab.stoi[text.unk_token]

    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
    if load:
        model.load_state_dict(torch.load("models/lstm.bin"))
    return model, fields


def train_model():
    model,fields = build_model()
    # load the dataset in json format
    train_ds, = data.TabularDataset.splits(
       path = 'data',
       train = 'train.csv',
       format = 'csv',
       fields = fields,
       skip_header = True
        )
    valid_ds, =  data.TabularDataset.splits(
       path = 'data',
       validation = 'eval.csv',
       format = 'csv',
       fields = fields,
       skip_header = True
        )


    train_iterator = data.BucketIterator(
        train_ds,
        batch_size=256,
        device=device, shuffle=True)
    valid_iterator = data.BucketIterator(valid_ds, batch_size=256, device=device, shuffle=False)



    optimizer = optim.Adam(model.parameters())
    loss_function = nn.NLLLoss().to(device)

    model.to(device)

    for i in range(100):
        train_loss, train_acc = train(model, train_iterator, optimizer, loss_function)
        eval_loss, eval_acc, l1_loss = evaluate(model, valid_iterator, loss_function)
        print(f"Epoch {i}, Train acc: {train_acc}, Train loss: {train_loss}, Eval acc {eval_acc}, Eval loss {eval_loss}, L1 loss: {l1_loss}")

    torch.save(model.state_dict(), "models/lstm.bin")

def test():
    model, fields = build_model(True)
    model.to(device)
    ds, =  data.TabularDataset.splits(
       path = ".",
        test=sys.argv[1],
       format = 'csv',
       fields = fields,
       skip_header = True
        )


    ds_iterator = data.BucketIterator(
        ds,
        batch_size=256,
        device=device, shuffle=False)
    predictions = []

    targets = []
    for batch in ds_iterator:
        p = model(*batch.reviewText)

        p = p.argmax(-1).detach().cpu().numpy()
        predictions.extend(p)
        targets.extend(batch.score.cpu().numpy())

    print("ACC", np.mean(np.array(predictions) == np.array(targets)))
    print("L1", np.mean(np.abs(np.array(predictions) - np.array(targets))))
    plot_confusion_matrx(predictions, targets)

if __name__ == "__main__":
    # if len(sys.argv) == 2:
    #     test()
    # else:
        train_model()
        test()
