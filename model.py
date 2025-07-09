import torch
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RFC():
    """
    Random Forest Classifier from sklearn
    """
    def __init__(self, n_estimators, seed):
        self.model = RandomForestClassifier(n_estimators, random_state = seed)
    
    def fit(self, X, y):
        """
        train the model with pytorch tensor X and y
        """
        self.model.fit([a.reshape(-1) for a in X.detach().numpy()], y.detach().numpy().reshape(-1))

    # def pytorch_predict(self, X):
    #     """
    #     predict with pytorch tensor X
    #     """
    #     result = []
    #     for X_ind in range(X.shape[0]):
    #         result.append(self.model.predict([[a.reshape(-1) for a in X.detach().numpy()][X_ind]]))
    #     result = np.array(result)
    #     result = torch.from_numpy(result)
    #     return result
    
    def predict(self, X):
        np_X = X.detach().numpy().reshape(X.shape[0], X.shape[1] *  X.shape[2])
        result = self.model.predict(np_X)
        result = torch.from_numpy(result)
        return result.reshape(X.shape[0], 1)

    def __str__(self) -> str:
        return "RF"
    

def rfc_predict_proba(rfc_model, X):
    """
    predict prob with pytorch tensor X
    """
    np_X = X.detach().numpy().reshape(X.shape[0], X.shape[1] *  X.shape[2])
    result = rfc_model.model.predict_proba(np_X)
    result = torch.from_numpy(result)

    #result = []
    #for X_ind in range(X.shape[0]):
        #result.append(rfc_model.model.predict_proba([[a.reshape(-1) for a in X.detach().numpy()][X_ind]]))
    #result = np.array(result)
    #result = torch.from_numpy(result)
    return result

import torch.nn as nn

class CNN_classifier(nn.Module):
    """
    CNN
    """
    def __init__(self, window_size, conv_out, f0, f1, f2, out):
        super(CNN_classifier,self).__init__()
        self.conv1d = nn.Conv1d(window_size, conv_out, kernel_size = 2)
        self.maxpool1d = nn.MaxPool1d(2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fc0 = nn.Linear(f0, 1)
        self.fc1 = nn.Linear(f1, f2)
        self.fc2 = nn.Linear(f2, out)
        
    def forward(self, X):
        out = self.conv1d(X)
        out = self.relu(out)
        out = self.maxpool1d(out)
        out = self.fc0(out)
        out = out[:, :, -1]
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

    def __str__(self) -> str:
        return "CNN"

import time
def train_classifier(model, lr, num_epochs, x_train, y_train, device, print_time = False):
    """
    train pytorch ANN models
    """
    model = model.to(device)
    model.train()
    #criterion = torch.nn.BCEWithLogitsLoss(reduction = "mean")
    criterion = torch.nn.BCELoss(reduction = "mean")
    optimiser = torch.optim.Adam(model.parameters(), lr = lr)
    start_time = time.time()
    losses = np.zeros(num_epochs)

    acc = 0
    epoch = 0
    #while acc <= 0.9 or epoch <= num_epochs:
    for epoch in range(num_epochs):
        
        y_train_pred = model(x_train.to(device))
        loss = criterion(y_train_pred, y_train.to(device))
        losses[epoch] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        #y_train_pred = model(x_train)
        #loss = criterion(y_train_pred, y_train)
        #y_train_pred = np.round(y_train_pred.detach())
        #acc = torch.sum(y_train_pred == y_train).item() / y_train_pred.shape[0]
        
    cost_time = time.time() - start_time

    if print_time:
        print("Training time of " + str(model) + " is: " + str(cost_time) + " Sec.")
        
    return losses

def evaluate_classifier(model, X_valid, y_valid, device, threshold):
    cnn_pred_valid = model(X_valid.to(device)).cpu()
    cnn_pred_valid_rounded = np.where(cnn_pred_valid.detach().numpy() > threshold, 1, 0)
    # return accuracy
    return torch.sum(torch.from_numpy(cnn_pred_valid_rounded) == y_valid).item() / y_valid.shape[0]

def grid_search_cnn_control(X_train, X_valid, y_train, y_valid, device, threshold):
    max_epochs = [100, 250, 500]
    learning_rate = [0.001, 0.025 ,0.01 ,0.005]
    w_s = X_train.shape[1]

    module__conv_out_and_f1 = [64, 128, 256]
    module__f0 = (X_train.shape[2] - 1)//2
    #module__f1 = [64, 128, 256, 512]
    module__f2 = [64, 128, 256]
    out = y_train.shape[1]

    best_score = -np.Inf
    best_cnn_paras = []
    for epoch in max_epochs:
        for lr in learning_rate: 
            for c in module__conv_out_and_f1:
                f1 = c
                for f2 in module__f2:
                    model = CNN_classifier(w_s, c, module__f0, f1, f2, out).to(device)
                    losses = train_classifier(model, lr, epoch, X_train, y_train, device)
                    score = evaluate_classifier(model, X_valid, y_valid, device, threshold)
                    if score > best_score:
                        # Better score accuracy indicates better performance
                        best_score = score
                        best_cnn_paras = [w_s, c, module__f0, f1, f2, out, epoch, lr]
        if int(best_score) == 1:
            break
    
    final_model = CNN_classifier(best_cnn_paras[0], best_cnn_paras[1],best_cnn_paras[2], best_cnn_paras[3], best_cnn_paras[4], best_cnn_paras[5]).to(device)
    final_losses = train_classifier(final_model, best_cnn_paras[-1], best_cnn_paras[-2], torch.cat((X_train, X_valid), 0), torch.cat((y_train, y_valid)), device)

    return final_model, final_losses

def grid_search_rnn(X_train, y_train, X_valid, y_valid, model_name, device, seed):
    torch.manual_seed(seed)
    max_epochs = [200]
    learning_rate = [0.001, 0.005, 0.01]
    #module__window_size = window_size
    input_dim = X_train.shape[2]
    module__hidden_dim = [32, 64]
    module__num_layers = [1, 2]
    output_dim = y_train.shape[1]

    best_score = np.Inf
    best_rnn_paras = []
    for epoch in max_epochs:
        for lr in learning_rate: 
            for hidden_dim in module__hidden_dim:
                for num_layers in module__num_layers:
                    if model_name == "LSTM":
                        model = LSTM(input_dim, hidden_dim, num_layers, output_dim)
                    elif model_name == "GRU":
                        model = GRU(input_dim, hidden_dim, num_layers, output_dim)
                    train_regressor(model, lr, epoch, X_train, y_train, device)
                    score = np.mean(evaluate_regressor(model, X_valid, y_valid, device)**2)
                    if score < best_score:
                        # Smaller score MSE indicates better performance
                        best_score = score
                        best_rnn_paras = [epoch, lr, input_dim, hidden_dim, num_layers, output_dim]

    if model_name == "LSTM":
        final_model = LSTM(best_rnn_paras[2], best_rnn_paras[3], best_rnn_paras[4], best_rnn_paras[5])
    elif model_name == "GRU":
        final_model = GRU(best_rnn_paras[2], best_rnn_paras[3], best_rnn_paras[4], best_rnn_paras[5])
    final_losses = train_regressor(final_model, best_rnn_paras[1], best_rnn_paras[0], torch.cat((X_train, X_valid), 0), torch.cat((y_train, y_valid), 0), device)

    return final_model, final_losses

#import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out
    
    def __str__(self) -> str:
        return "GRU"

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

    def __str__(self) -> str:
        return "LSTM"
    
#import time
#import numpy as np
#import torch

def train_regressor(model, lr, num_epochs, x_train, y_train, device, print_time = False):
    """
    train pytorch ANN forecasting models
    """
    model = model.to(device)
    model.train()
    criterion = torch.nn.MSELoss(reduction = "mean")
    optimiser = torch.optim.Adam(model.parameters(), lr = lr)
    start_time = time.time()
    losses = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        y_train_pred = model(x_train.to(device))
        loss = criterion(y_train_pred, y_train.to(device))
        losses[epoch] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    cost_time = time.time() - start_time

    if print_time:
        print("Training time of " + str(model) + " is: " + str(cost_time) + " Sec.")
        
    return losses 

def evaluate_regressor(model, X, y, device):
    """
    return error array
    """
    if str(model) != "RF":
        model.eval()
    return (predict_regressor(model.to(device), X.to(device), device) - y.to(device)).cpu().detach().numpy()

def predict_regressor(model, X, device):
    if str(model) == "RF":
        result = model.rf_pytorch_predict(X)
    else:
        model.eval()
        result = model(X.to(device))
        
    return result

class TransformerModel(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super(TransformerModel, self).__init__()

        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x[:, -1, :])
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)