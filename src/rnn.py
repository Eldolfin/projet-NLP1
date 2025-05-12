import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import nltk
from collections import Counter
from utils import Prediction
import itertools
import time

#Model definition
  
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional=False):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, bidirectional=bidirectional, batch_first=True, num_layers=n_layers, dropout=0.5)
        self.h2o = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax()

    def forward(self, text):
        rnn_output, hidden = self.rnn(text)
        out = self.h2o(hidden[0])
        out = self.softmax(out)
        return out
    

#private functions
def get_index(traindata, validdata):
    X_train_tokenized = [nltk.word_tokenize(x) for x in traindata]
    X_valid_tokenized = [nltk.word_tokenize(x) for x in validdata]

    #Create the word corpus to create embeddings
    corpus = Counter(list(itertools.chain(*X_train_tokenized)))
    corpus = sorted(corpus,key=corpus.get,reverse=True)
    onehot_dict = {w:i+1 for i,w in enumerate(corpus)}

    #Create the embeddings
    X_train_embeddings = [[onehot_dict[word] for word in sentence] for sentence in X_train_tokenized]
    #We use an abritrary value for unk words (last index) -> no info from these words
    X_valid_embeddings = [[onehot_dict[word] if word in onehot_dict else len(onehot_dict) for word in sentence] for sentence in X_valid_tokenized]
    return X_train_embeddings, X_valid_embeddings, onehot_dict

def get_vocab_size(X_train):
    X_train_tokenized = [nltk.word_tokenize(x) for x in X_train]
    corpus = Counter(list(itertools.chain(*X_train_tokenized)))
    return len(corpus)

vocab_size = None

def _train(model, training_data, n_epoch, learning_rate, report_every = 0, criterion = nn.CrossEntropyLoss(), optimizer = torch.optim.SGD):
    current_loss = 0
    model.train()
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

    for iter in range(1, n_epoch + 1):
        model.zero_grad() # clear the gradients

        for sentences, labels in training_data:
            input_tensor = sentences
            output = model.forward(input_tensor.float())
            loss = criterion(output, labels)
            # optimize parameters
            loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()

            current_loss += loss.item()
        
        scheduler.step(current_loss)
        
        if report_every != 0 and iter % report_every == 0:
            print(f"{iter} ({iter / n_epoch:.0%}): \t average batch loss = {current_loss / len(training_data)}")
        current_loss = 0

def train(X_train, Y_train, X_test, Y_test):
    #Hyperparams:
    batch_size = 25

    #load data
    vocab_size = get_vocab_size(X_train)


    def collate_pack_onehot(batch):
        data = [nn.functional.one_hot(item[0], num_classes=vocab_size + 1).float() for item in batch]
        packed_data = nn.utils.rnn.pack_sequence(data, enforce_sorted=False)
        target = [item[1] for item in batch]
        return packed_data, torch.tensor(target)
    
    embedded_train, embedded_valid, onehot_dict = get_index(X_train, X_test)
    train_tensors = [torch.tensor(x) for x in embedded_train]
    train_tensors = list(zip(train_tensors, torch.tensor(Y_train)))
    valid_tensors = [torch.tensor(x) for x in embedded_valid]
    valid_tensors = list(zip(valid_tensors, torch.tensor(Y_test)))
    trainloader = DataLoader(train_tensors, collate_fn=collate_pack_onehot, shuffle=True, batch_size=batch_size)
    validloader = DataLoader(valid_tensors, collate_fn=collate_pack_onehot, batch_size=batch_size)

    model = RNN(vocab_size + 1, 500, 18, 3)
    _train(model, trainloader, 100, 1e-4, optimizer=torch.optim.Adam)
    return model, onehot_dict

def classify(ds, model, onehot_dict, input, method):
    before = time.process_time()
    vocab_size = len(onehot_dict)
    scenario_decoder = ds["train"].features["scenario"].int2str

    tokens = nltk.word_tokenize(input)
    indexes = [onehot_dict[word] for word in tokens if word in onehot_dict]
    if len(indexes) == 0:
        return Prediction(
            method, "No guess", "", 0, before=before
        )
    embedded = nn.functional.one_hot(torch.LongTensor(indexes), num_classes=vocab_size + 1).float()
    model.eval()
    output = model.forward(embedded)
    index = output.argmax().item()
    proba = output[index].item()

    return Prediction(
        method, scenario_decoder(index), "", proba, before=before
    )