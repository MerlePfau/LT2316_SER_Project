import os
import torch
import torch.nn as nn
from torch import optim
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
import torch.nn.functional as F

class Batcher:
    def __init__(self, X, y, batch_size=50, max_iter=None):
        self.X = X
        self.y = y
        #self.device = device
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.curr_iter = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.curr_iter == self.max_iter:
            raise StopIteration
        permutation = torch.randperm(self.X.shape[0])#, device=self.device)
        permX = self.X[permutation]
        permy = self.y[permutation]
        splitX = torch.split(permX, self.batch_size)
        splity = torch.split(permy, self.batch_size)
        
        self.curr_iter += 1
        return zip(splitX, splity)


class LSTM_fixed_len(nn.Module) :
    def __init__(self, input_size, hidden_dim, output_size, n_layers=3) :
        super().__init__()
        
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=n_layers, dropout=0.2, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_size)    
        self.softmax = nn.LogSoftmax(dim=2)  
    
    def forward(self, x):#, device):
        lstm_out, (h, c) = self.lstm(x)    
        #lstm_out, h, c =  lstm_out.to(device), h.to(device), c.to(device)
        
        output = self.linear(h[-1])
        return output

    
class Trainer:

    def __init__(self, device, dump_folder="/tmp/aa2_models/"):
        self.device = device
        self.dump_folder = dump_folder
        os.makedirs(dump_folder, exist_ok=True)
        
        
    def save_model(self, epoch, model, optimizer, batch_size, learning_rate, hidden, number_lay, loss, scores, model_name):
        # epoch = epoch
        # model =  a train pytroch model
        # optimizer = a pytorch Optimizer
        # loss = loss (detach it from GPU)
        # scores = dict where keys are names of metrics and values the value for the metric
        # hyperparamaters = dict of hyperparamaters
        # model_name = name of the model
        save_dict = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'hidden_dim': hidden,
                        'num_layers': number_lay,
                        'loss': loss,
                        'scores': scores,
                        'model_name': model_name
                        }

        torch.save(save_dict, os.path.join(self.dump_folder, model_name + ".pt"))

    def load_model(self, model_path):
        # Finish this function so that it loads a model and return the appropriate variables
        checkpoint = torch.load(model_path)
        epoch = checkpoint['epoch']
        model_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        batch_size = checkpoint['batch_size']
        learning_rate = checkpoint['learning_rate']
        hidden = checkpoint['hidden_dim']
        num_lay = checkpoint['num_layers']
        loss = checkpoint['loss']
        scores = checkpoint['scores']
        model_name = checkpoint['model_name']

        return epoch, model_state_dict, optimizer_state_dict, batch_size, learning_rate, hidden, num_lay, loss, scores, model_name
    
        
    def train_model(self, model, x_train, y_train, x_val, y_val, hp, output_size, epochs=100):
        self.output_size = output_size
        m = model(input_size=x_train.shape[2], hidden_dim=hp["hidden_size"], output_size=self.output_size, n_layers=hp["number_layers"])
        m = m.to(self.device)
        model_name=hp["model"]
        batch_size=hp['batch_size']
        lr=hp["learning_rate"]
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(m.parameters(), lr=lr)
        b = Batcher(x_train, torch.LongTensor(y_train), batch_size, max_iter=epochs)
        e = 0
        for split in b:
            m.train()
            tot_loss = 0
            for X, y in split:
                #X = X.unsqueeze(dim=1)
                optimizer.zero_grad()
                o = m(X.float())#, device)
                l = criterion(o, y)#.to(device)
                tot_loss += l
                l.backward()
                optimizer.step()    
            m.eval()

            bl = Batcher(x_val, y_val, batch_size, max_iter=1)
            y_true = []
            y_pred = []
            for split in bl:
                for X, y in split:
                    #X = X.unsqueeze(dim=1)
                    predictions = m(X.float())
                    labels = y
                    for i in range(predictions.shape[0]):
                        label = labels[i]
                        prediction = tf.argmax(input=predictions[i].tolist())
                        y_true.append(label)
                        y_pred.append(prediction)
            scores = {}
            f = f1_score(y_true, y_pred, average='weighted')
            scores['f1_score'] = f
            print("{}: Total loss in epoch {} is: {}      |      F1 score in validation is: {}".format(model_name, e, tot_loss, f), end='\r')
            e += 1
        print('\n Finished training {}.'.format(model_name))
        self.save_model(e, m, optimizer, batch_size, lr, hp["hidden_size"], hp["number_layers"], tot_loss, scores, model_name)
        pass

    
    def predict(self, x_test, model_class, best_model_path):
        trained_epochs, model_state_dict, optimizer_state_dict, trained_batch_size, trained_learning_rate, trained_hidden_dim, trained_num_lay, trained_loss, trained_scores, model_name = self.load_model(best_model_path)
        
        m = model_class(x_test.shape[2], trained_hidden_dim, self.output_size, n_layers=trained_num_lay)
        m.load_state_dict(model_state_dict)
        m = m.to(self.device)
        m.eval()
        
        X = x_test

        #X = x_test.unsqueeze(dim=1)
        predictions = m(X.float())
        y_pred = []
        for i in range(predictions.shape[0]):
            prediction = tf.argmax(input=predictions[i].tolist())
            y_pred.append(prediction)

        return y_pred
    