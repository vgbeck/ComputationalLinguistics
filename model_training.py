import torch
import torch.nn as nn
from model import Emb_RNN
import numpy as np
import re
import sys
import collections
import os
import random
import json

verbose = False

num_epochs = 1


d_emb = 64
n_layers = 1
d_hid = 64
lr = 0.0003
use_LSTM = True
if use_LSTM:
    model_type = 'lstm'
else:
    model_type = 'rnn'

def train(net, lines, params):
    criterion = nn.CrossEntropyLoss() #Don't use ignore index!!!
    optimiser = torch.optim.Adam(net.parameters(), lr=lr)
    if os.path.exists(params['save_path']):
        checkpoint = torch.load(params['save_path'])
        print('Loading checkpoint')
        net.load_state_dict(checkpoint['net_state_dict'])
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        net.eval()


    for epoch in range(1):
        ep_loss = 0.
        num_tested = 0
        num_correct = 0
        for counter, i in enumerate(torch.randperm(len(lines))):
        #for counter, i in enumerate(torch.randperm(20)):
            line = torch.LongTensor([wd2ix[wd] for wd in lines[i][0]])
            pred = net(line)
            pred = pred.contiguous().view(-1, pred.size(-1))
            target = torch.tensor(au2ix[lines[i][1]])
            target = target.contiguous().view(-1)
            target = target.long()
            with torch.no_grad():
                pred_numpy = np.argmax(pred.numpy(), axis=1).tolist()
                target_numpy = target.numpy().tolist()
                num_tested += 1
                if pred_numpy == target_numpy:
                    num_correct += 1
            loss = criterion(pred, target)
            if torch.isnan(loss):
                with torch.no_grad():
                    print(pred, target, lines[i])
                    exit()
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            ep_loss += loss.detach() 
        print('Epoch', epoch, 'Accuracy', round(num_correct / num_tested, 4), 'Loss', ep_loss)
        print('Saving checkpoint')
        torch.save({'net_state_dict': net.state_dict(),  'optimiser_state_dict': optimiser.state_dict()}, params['save_path'])



def test(net, lines, params):
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(net.parameters(), lr=lr)
    if os.path.exists(params['save_path']):
        checkpoint = torch.load(params['save_path'])
        print('Loading checkpoint')
        net.load_state_dict(checkpoint['net_state_dict'])
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        net.eval()
    num_tested = 0
    num_correct = 0
    with torch.no_grad():
        current_author = None
        scores = np.zeros(len(authors))
        for line in lines:
            wds = torch.LongTensor([wd2ix[wd] for wd in line[0]])
            pred = net(wds)
            pred = pred.contiguous().view(-1, pred.size(-1))
            target = torch.tensor(au2ix[line[1]])
            target = target.contiguous().view(-1)
            target = target.long()
            pred_numpy = np.argmax(pred.numpy(), axis=1).tolist()
            target_numpy = target.numpy().tolist()
            num_tested += 1
            if pred_numpy == target_numpy:
                num_correct += 1
            author = au2ix[line[1]]
            if author != current_author:
                #We have moved to a new author so we want to collect the scores of the most recent author
                if current_author is not None:
                    #If we are not at the very beginning
                    most_frequent_author = ix2au[str(np.argmax(scores))] #This author was predicted the most of all the lines of the most recent current author
                    print(ix2au[str(current_author)], most_frequent_author) #Does the current author match the most frequently predicted author?
                    print(scores)
                    print(scores[current_author]) #How many hits did the current author actually get?
                    scores = np.zeros(len(authors)) #Reset the scores to zero for each author
                current_author = author #Reset the current author to the most recent author
            scores[pred_numpy[0]] += 1 #Add 1 to the score of the predicted author for the last lline seen
    print('Test accuracy', round(num_correct / num_tested, 4))
 

#net = Emb_RNN()
vocab = []
authors = []
train_lines = []
test_lines = []

with open('trainfile-1.json', 'r') as f0:
    train_data = json.load(f0)
    for pair in train_data:
        line = pair[0].split()
        line = [re.sub(r'[^a-zA-Z\*’\']', '', wd.lower()) for wd in line]
        for wd in line:
            wd = re.sub(r'[^a-zA-Z\*’\']', '', wd)
            if wd.lower() not in vocab:
                vocab.append(wd.lower())
        if pair[1] not in authors:
            authors.append(pair[1])
        train_lines.append([line, pair[1]])
        
with open('testfile-1.json', 'r') as f0:
    test_data = json.load(f0)
    for pair in test_data:
        line = pair[0].split()
        line = [re.sub(r'[^a-zA-Z\*’\']', '', wd.lower()) for wd in line]
        for wd in line:
            wd = re.sub(r'[^a-zA-Z\*’\']', '', wd)
            if wd.lower() not in vocab:
                vocab.append(wd.lower())
        if pair[1] not in authors:
            authors.append(pair[1])
        test_lines.append([line, pair[1]])
        

print('There are', len(vocab), 'words in the vocabulary') 
print('There are', len(authors), 'authors') 
print(authors)

wd2ix = {}
ix2wd = {}
for i,wd in enumerate(vocab):
    wd2ix[wd] = i
    ix2wd[str(i)] = wd
#words_as_indices = [torch.LongTensor([wd2ix[wd] for wd in vocab])]


au2ix = {}
ix2au = {}
for i,au in enumerate(authors):
    au2ix[au] = i
    ix2au[str(i)] = au
#authors_as_indices = [torch.LongTensor([au2ix[au] for au in authors])]

print(au2ix.items())

params = {'num_wds': len(vocab), 'num_authors': len(authors), 'd_emb': 128, 'num_layers': 1, 'd_hid': 128, 'lr': 0.0003, 'epochs': 5, 'save_path': 'authors.pth'}

model = Emb_RNN(params, True)
for j in range(1):
    train(model, train_lines, params)
    test(model, test_lines, params)


