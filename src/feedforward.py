import torch
from datasets import load_dataset
import nltk
from collections import Counter
import itertools
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
from gensim.models import Word2Vec
import gensim.downloader as api

#Model definition

class Feedforward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation_function):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.6)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation_function = activation_function
    
    def forward(self, text):
        hidden = self.fc1(text.float())
        hidden = self.activation_function(hidden)

        hidden = self.dropout(hidden)
        out = self.fc2(hidden)
        return out

#private functions
def _train(model, training_data, n_epoch, learning_rate, report_every = 1, criterion = nn.CrossEntropyLoss(), optimizer = torch.optim.SGD):
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

def train(ds, X_train, Y_train, X_test, Y_test):
    #Hyperparams:
    batch_size = 25

    #load data
    vectorizer = TfidfVectorizer()
    X_train_tfidf = torch.Tensor(vectorizer.fit_transform(X_train).toarray())
    X_test_tfidf = torch.Tensor(vectorizer.transform(X_test).toarray())
    train_data_tfidf = TensorDataset(X_train_tfidf, torch.tensor(Y_train))
    test_data_tfidf = TensorDataset(X_test_tfidf, torch.tensor(Y_test))
    trainloader = DataLoader(train_data_tfidf, shuffle=True, batch_size=batch_size)
    validloader = DataLoader(test_data_tfidf, batch_size=batch_size)

    model = Feedforward(len(trainloader.dataset[0][0]), 100, 18, nn.functional.relu)
    _train(model, trainloader, 100, 1e-4, optimizer=torch.optim.Adam)
    return model