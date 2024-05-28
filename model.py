import torch
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RF():
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
    

def rf_predict_proba(rfc_model, X):
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
def train_classifier(model, lr, num_epochs, x_train, y_train, print_time = False):
    """
    train pytorch ANN models
    """

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
        
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train)
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

def evaluate(model, X_valid, y_valid, threshold):
    cnn_pred_valid = model(X_valid).cpu()
    cnn_pred_valid_rounded = np.where(cnn_pred_valid.detach().numpy() > threshold, 1, 0)
    # return accuracy
    return torch.sum(torch.from_numpy(cnn_pred_valid_rounded) == y_valid).item() / y_valid.shape[0]

def grid_search_cnn_control(X_train, X_valid, y_train, y_valid, threshold):
    max_epochs = [100, 250]
    learning_rate = [0.025 ,0.01 ,0.005]
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
                    model = CNN_classifier(w_s, c, module__f0, f1, f2, out)
                    losses = train_classifier(model, lr, epoch, X_train, y_train)
                    score = evaluate(model, X_valid, y_valid, threshold)
                    if score > best_score:
                        # Better score accuracy indicates better performance
                        best_score = score
                        best_cnn_paras = [w_s, c, module__f0, f1, f2, out, epoch, lr]
        if int(best_score) == 1:
            break
    
    final_model = CNN_classifier(best_cnn_paras[0], best_cnn_paras[1],best_cnn_paras[2], best_cnn_paras[3], best_cnn_paras[4], best_cnn_paras[5])
    final_losses = train_classifier(final_model, best_cnn_paras[-1], best_cnn_paras[-2], torch.cat((X_train, X_valid), 0), torch.cat((y_train, y_valid)))

    return final_model, final_losses