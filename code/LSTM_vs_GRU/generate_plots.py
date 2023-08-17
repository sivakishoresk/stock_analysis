

import numpy as np
import pandas as pd
import os

names = ['GOOG','AAPL','MSFT','TATAMOTORS.NS','BTC-USD']
for name in names:
    data = pd.read_csv('./datasets/{}.csv'.format(name))
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values('Date')
    price = data['Close']
    data.head()
    path = './{}'.format(name)
    os.makedirs(path)

    import matplotlib.pyplot as plt
    import seaborn as sns


    days_number = 400
    sns.set_style("darkgrid")
    plt.figure(figsize = (15,9))
    plt.plot(data['Close'])
    plt.xticks(range(0,data.shape[0],days_number),data['Date'].loc[::days_number],rotation=45)
    plt.title("{} Stock Price".format(name),fontsize=18, fontweight='bold')
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Close Price (USD)',fontsize=18)
    plt.savefig("./{}/{}_total_stocks.jpg".format(name,name))
    #plt.show()

    #Normalization of data
    price = data[['Close']]
    price.info()

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(-1, 1))
    price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))
    def split_data(stock, lookback):
        data_raw = stock.to_numpy() # convert to numpy array
        data = []
        
        # create all possible sequences of length seq_len
        for index in range(len(data_raw) - lookback): 
            data.append(data_raw[index: index + lookback])
        
        data = np.array(data);
        test_set_size = int(np.round(0.2*data.shape[0]));
        train_set_size = data.shape[0] - (test_set_size);
        
        x_train = data[:train_set_size,:-1,:]
        y_train = data[:train_set_size,-1,:]
        
        x_test = data[train_set_size:,:-1]
        y_test = data[train_set_size:,-1,:]
        
        return [x_train, y_train, x_test, y_test]

    lookback = 30 # choose sequence length
    x_train, y_train, x_test, y_test = split_data(price, lookback)
    print('x_train.shape = ',x_train.shape)
    print('y_train.shape = ',y_train.shape)
    print('x_test.shape = ',x_test.shape)
    print('y_test.shape = ',y_test.shape)

    input_dim = 1
    hidden_dim = 32
    num_layers = 2
    output_dim = 1
    num_epochs = 100

    import torch
    import torch.nn as nn

    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
    y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)
    y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
    y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)

    #LSTM model
    class LSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
            super(LSTM, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
            out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
            out = self.fc(out[:, -1, :]) 
            return out

    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    import time

    hist = np.zeros(num_epochs)
    start_time = time.time()
    lstm = []

    for t in range(num_epochs):
        y_train_pred = model(x_train)

        loss = criterion(y_train_pred, y_train_lstm)
        print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
    training_time = time.time()-start_time
    print("Training time: {}".format(training_time))

    predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
    original = pd.DataFrame(scaler.inverse_transform(y_train_lstm.detach().numpy()))

    import seaborn as sns
    sns.set_style("darkgrid")    

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    plt.subplot(1, 2, 1)
    ax = sns.lineplot(x = original.index, y = original[0], label="Data", color='royalblue')
    ax = sns.lineplot(x = predict.index, y = predict[0], label="Training Prediction (LSTM)", color='tomato')
    ax.set_title('.{} LSTM train Stock price'.format(name), size = 14, fontweight='bold')
    ax.set_xlabel("Days", size = 14)
    ax.set_ylabel("Cost (USD)", size = 14)
    ax.set_xticklabels('',size = 14)



    plt.subplot(1, 2, 2)
    ax = sns.lineplot(data=hist, color='royalblue')
    ax.set_xlabel("Epoch", size = 14)
    ax.set_ylabel("Loss", size = 14)
    ax.set_title("Training Loss", size = 14, fontweight='bold')
    fig.set_figheight(6)
    fig.set_figwidth(16)
    plt.savefig('./{}/{}_LSTM_train_loss_prediction.jpg'.format(name,name))
    #plt.show()

    import math, time
    from sklearn.metrics import mean_squared_error

    # make predictions
    y_test_pred = model(x_test)

    # invert predictions
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train = scaler.inverse_transform(y_train_lstm.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test_lstm.detach().numpy())

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    lstm.append(trainScore)
    lstm.append(testScore)
    lstm.append(training_time)

    original = pd.DataFrame(y_test)
    predict = pd.DataFrame(y_test_pred)
    fig = plt.figure()

    ax = sns.lineplot(x = original.index, y = original[0], label="Data", color='royalblue')
    ax = sns.lineplot(x = predict.index, y = predict[0], label="Test Prediction (LSTM)", color='tomato')
    ax.set_title('{} LSTM Test Stock price'.format(name), size = 14, fontweight='bold')
    ax.set_xlabel("Days", size = 14)
    ax.set_ylabel("Cost (USD)", size = 14)
    ax.set_xticklabels('', size=10)
    fig.set_figheight(6)
    fig.set_figwidth(16)
    plt.savefig('./{}/{}_LSTM_test_prediction.jpg'.format(name,name))
    #plt.show()

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

    model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    hist = np.zeros(num_epochs)
    start_time = time.time()
    gru = []

    for t in range(num_epochs):
        y_train_pred = model(x_train)

        loss = criterion(y_train_pred, y_train_gru)
        print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    training_time = time.time()-start_time    
    print("Training time: {}".format(training_time))

    predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
    original = pd.DataFrame(scaler.inverse_transform(y_train_gru.detach().numpy()))

    import seaborn as sns
    sns.set_style("darkgrid")    

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    plt.subplot(1, 2, 1)
    ax = sns.lineplot(x = original.index, y = original[0], label="Data", color='royalblue')
    ax = sns.lineplot(x = predict.index, y = predict[0], label="Training Prediction (GRU)", color='tomato')
    ax.set_title('{} GRU Train Stock price'.format(name), size = 14, fontweight='bold')
    ax.set_xlabel("Days", size = 14)
    ax.set_ylabel("Cost (USD)", size = 14)
    ax.set_xticklabels('', size=10)


    plt.subplot(1, 2, 2)
    ax = sns.lineplot(data=hist, color='royalblue')
    ax.set_xlabel("Epoch", size = 14)
    ax.set_ylabel("Loss", size = 14)
    ax.set_title("Training Loss", size = 14, fontweight='bold')
    fig.set_figheight(6)
    fig.set_figwidth(16)
    plt.savefig("./{}/{}_GRU_train_loss.jpg".format(name,name))

    import math, time
    from sklearn.metrics import mean_squared_error

    # make predictions
    y_test_pred = model(x_test)

    # invert predictions
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train = scaler.inverse_transform(y_train_gru.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test_gru.detach().numpy())

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    gru.append(trainScore)
    gru.append(testScore)
    gru.append(training_time)

    original = pd.DataFrame(y_test)
    predict = pd.DataFrame(y_test_pred)
    fig = plt.figure()

    ax = sns.lineplot(x = original.index, y = original[0], label="Data", color='royalblue')
    ax = sns.lineplot(x = predict.index, y = predict[0], label="Test Prediction (LSTM)", color='tomato')
    ax.set_title('{} GRU TestStock price'.format(name), size = 14, fontweight='bold')
    ax.set_xlabel("Days", size = 14)
    ax.set_ylabel("Cost (USD)", size = 14)
    ax.set_xticklabels('', size=10)
    fig.set_figheight(6)
    fig.set_figwidth(16)
    plt.savefig('./{}/{}_GRU_test_prediction.jpg'.format(name,name))
    #plt.show()

    lstm = pd.DataFrame(lstm, columns=['LSTM'])
    gru = pd.DataFrame(gru, columns=['GRU'])
    result = pd.concat([lstm, gru], axis=1, join='inner')
    result.index = ['Train RMSE', 'Test RMSE', 'Train Time']
    result.to_csv('./{}/{}_errors.csv'.format(name,name))