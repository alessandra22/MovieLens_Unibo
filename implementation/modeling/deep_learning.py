import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import numpy as np
from torcheval.metrics.functional import r2_score
from dataset.data_visualization import draw_loss
from implementation.data_preprocessing import get_validation_split

PATH = 'models-65-15-20/'


def get_dataloader(X_train, X_val, y_train, y_val, batch_size, device):
    train_tensor = torch.tensor(np.float32(X_train))
    train_target_tensor = torch.tensor(np.float32(y_train.to_numpy()))
    train_target_tensor = train_target_tensor.reshape((train_target_tensor.shape[0], 1))
    val_tensor = torch.tensor(np.float32(X_val))
    val_target_tensor = torch.tensor(np.float32(y_val.to_numpy()))
    val_target_tensor = val_target_tensor.reshape((val_target_tensor.shape[0], 1))
    train_tensor = train_tensor.to(device)
    val_tensor = val_tensor.to(device)
    train_target_tensor = train_target_tensor.to(device)
    val_target_tensor = val_target_tensor.to(device)

    train_dataset = TensorDataset(train_tensor, train_target_tensor)
    val_dataset = TensorDataset(val_tensor, val_target_tensor)
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=True),
    )


class MyNet(nn.Module):
    def __init__(self, input_shape, n_hidden_layer, dim_hidden_layer):
        self.n_hidden_layer = n_hidden_layer
        super(MyNet, self).__init__()
        if n_hidden_layer > 0:
            self.hid1 = nn.Linear(input_shape[1], dim_hidden_layer)
        if n_hidden_layer > 1:
            self.hid2 = nn.Linear(dim_hidden_layer, dim_hidden_layer)
        if n_hidden_layer > 2:
            self.hid3 = nn.Linear(dim_hidden_layer, dim_hidden_layer)

        self.out = nn.Linear(dim_hidden_layer, 1)

    def forward(self, x):
        if self.n_hidden_layer > 0:
            x = torch.relu(self.hid1(x))
        if self.n_hidden_layer > 1:
            x = torch.relu(self.hid2(x))
        if self.n_hidden_layer > 2:
            x = torch.relu(self.hid3(x))
        x = self.out(x)
        return x


def train_model(X_train, X_val, y_train, y_val, bs, lf, ne, nhl, dhl, lr, debug):
    if bs > 16:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'

    train_dl, valid_dl = get_dataloader(X_train, X_val, y_train, y_val, bs, device)
    net = MyNet(X_train.shape, nhl, dhl)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    min_valid_loss = np.inf
    best_model = []
    hist_train_loss = []
    hist_val_loss = []

    for epoch in range(ne):
        net.train(True)
        train_loss = 0
        for batch, (X, y) in enumerate(train_dl):
            optimizer.zero_grad()
            pred = net(X)
            loss = lf(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dl)
        hist_train_loss.append(train_loss)
        if debug:
            print(f'Train loss at epoch {epoch+1}: {train_loss}')

        net.eval()
        valid_loss = 0
        with torch.no_grad():
            for data, labels in valid_dl:
                target = net(data)
                loss = lf(target, labels)
                valid_loss += loss.item()

            valid_loss /= len(valid_dl)
            hist_val_loss.append(valid_loss)
            if debug:
                print(f'Val loss at epoch {epoch+1}: {valid_loss}')

            if min_valid_loss > valid_loss:
                if debug:
                    print('--------------val loss improved: saving the state of the model')
                min_valid_loss = valid_loss
                best_model = net.state_dict()

    torch.save(best_model, PATH + f'bs-{bs}_' + f'ne-{ne}_' + f'hl-{nhl}x{dhl}_' + f'lr-{lr}' + '.pth')
    return [hist_train_loss, hist_val_loss]


def test_model(X_train, X_val, X_test, y_train, y_val, y_test, bs, ne, nhl, dhl, lr, debug):
    if bs > 16:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    test_input_tensor = torch.tensor(X_test).to(torch.float32)
    test_target_tensor = torch.tensor(y_test.values).to(torch.float32)
    test_target_tensor = test_target_tensor.reshape((test_target_tensor.shape[0], 1))
    test_input_tensor = test_input_tensor.to(device)
    test_target_tensor = test_target_tensor.to(device)
    mse = nn.MSELoss()

    print(f'Training network with: batch_size of {bs} in {ne} epochs, {nhl}x{dhl} and learning rate = {lr}...')
    [train_loss, val_loss] = train_model(X_train, X_val, y_train, y_val, bs, nn.MSELoss(), ne, nhl, dhl, lr, debug)
    print('Testing best network with test dataset...')
    best_model = MyNet(X_train.shape, nhl, dhl)
    best_model.load_state_dict(torch.load(PATH + f'bs-{bs}_' + f'ne-{ne}_' + f'hl-{nhl}x{dhl}_' + f'lr-{lr}' + '.pth'))
    best_model.eval()
    best_model = best_model.to(device)
    test_output_tensor = best_model(test_input_tensor)
    my_mse = mse(test_output_tensor, test_target_tensor)
    my_r2 = r2_score(test_output_tensor, test_target_tensor)
    print(f'MSE: {my_mse}\nR2 score: {my_r2}\n\n')
    return [my_mse, my_r2, train_loss, val_loss]


def deep_test(X_df, y_df, ts, vs, grid=True, debug=False):
    X_train, X_val, X_test, y_train, y_val, y_test = get_validation_split(X_df, y_df, ts, vs)

    if grid:
        if vs == 15:
            res = test_model(X_train, X_val, X_test, y_train, y_val, y_test, 32, 80, 1, 8, 0.001, debug)
            print(f'MSE: {res[0]}\nR2 score: {res[1]}')
            draw_loss(res[2], res[3], '15-20')
            return res
        else:
            res = test_model(X_train, X_val, X_test, y_train, y_val, y_test, 32, 80, 1, 8, 0.001, debug)
            print(f'MSE: {res[0]}\nR2 score: {res[1]}')
            draw_loss(res[2], res[3], '10-20')
            return res
    else:

        params = {
            'n_hidden_layers': [1, 2, 3],
            'dim_hidden_layers': [8, 16, 32, 64],
            'batch_size': [16, 32, 64, 128, 256],
            'n_epochs': [10, 20, 50, 80],
            'learning_rate': [0.01, 0.001, 0.0001]
        }

        results = dict()

        for nhl in params['n_hidden_layers']:
            for dhl in params['dim_hidden_layers']:
                for bs in params['batch_size']:
                    for ne in params['n_epochs']:
                        for lr in params['learning_rate']:
                            res = test_model(X_train, X_val, X_test, y_train, y_val, y_test, bs, ne, nhl, dhl, lr, debug)
                            key = f'bs-{bs}_' + f'ne-{ne}_' + f'opt-Adam_' + f'hl-{nhl}x{dhl}_' + f'lr-{lr}'
                            results[key] = dict()
                            results[key]['mse'] = res[0]
                            results[key]['r2'] = res[1]
                            results[key]['train_loss'] = res[2]
                            results[key]['val_loss'] = res[3]

        best_mse = np.inf
        best_mse_key = []
        best_r2 = 0
        best_r2_key = []

        for k in results.keys():
            new_mse = float(results[k]["mse"])
            new_r2 = float(results[k]["r2"])
            if new_mse == best_mse:
                best_mse_key.append(k)
            elif new_mse < best_mse:
                best_mse = new_mse
                best_mse_key = [k]

            if new_r2 == best_r2:
                best_r2_key.append(k)
            elif new_r2 > best_r2:
                best_r2 = new_r2
                best_r2_key = [k]

            print(f'{k}: mse={round(float(new_mse), 3)}, r2={round(float(new_r2), 3)}')

        print(f'best mse = {best_mse} in {best_mse_key}')
        print(f'best r2 = {best_r2} in {best_r2_key}')
        return results
