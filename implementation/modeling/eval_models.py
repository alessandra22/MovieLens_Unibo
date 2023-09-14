import torch
import torch.nn as nn
from torcheval.metrics.functional import r2_score
from dataset.data_acquisition import load_csv
from implementation.data_preprocessing import get_validation_split


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


if torch.cuda.is_available():
    device = torch.device("cuda")

PATH = '../../models-65-15-20/'
def test_model(X_train, X_test, y_test, bs, ne, nhl, dhl, lr):
    test_input_tensor = torch.tensor(X_test).to(torch.float32)
    test_target_tensor = torch.tensor(y_test.values).to(torch.float32)
    test_target_tensor = test_target_tensor.reshape((test_target_tensor.shape[0], 1))
    mse = nn.MSELoss()

    print(f'Testing network {nhl}x{dhl}, bs={bs}, ne={ne}, lr={lr}')
    best_model = MyNet(X_train.shape, nhl, dhl)
    best_model.load_state_dict(torch.load(PATH + f'bs-{bs}_' + f'ne-{ne}_' + f'hl-{nhl}x{dhl}_' + f'lr-{lr}' + '.pth'))
    best_model.eval()
    test_output_tensor = best_model(test_input_tensor)
    my_mse = mse(test_output_tensor, test_target_tensor)
    my_r2 = r2_score(test_output_tensor, test_target_tensor)
    print(f'MSE: {my_mse}\nR2 score: {my_r2}\n\n')
    return [my_mse, my_r2]


params = {
    'n_hidden_layers': [1, 2, 3],
    'dim_hidden_layers': [8, 16, 32, 64],
    'batch_size': [16, 32, 64, 128, 256],
    'n_epochs': [10, 20, 50, 80],
    'learning_rate': [0.01, 0.001, 0.0001]
}

results = dict()

clean_df = load_csv('./../../clean_df.csv')
clean_df.set_index('movieId', inplace=True)

y_df = clean_df['rating']
X_df = clean_df.drop('rating', axis=1)
X_train, X_val, X_test, y_train, y_val, y_test = get_validation_split(X_df, y_df, 0.20, 0.10)

for nhl in params['n_hidden_layers']:
    for dhl in params['dim_hidden_layers']:
        for bs in params['batch_size']:
            for ne in params['n_epochs']:
                for lr in params['learning_rate']:
                    res = test_model(X_train, X_test, y_test, bs, ne, nhl, dhl, lr)
                    key = f'bs-{bs}_' + f'ne-{ne}_' + f'opt-Adam_' + f'hl-{nhl}x{dhl}_' + f'lr-{lr}'
                    results[key] = dict()
                    results[key]['mse'] = res[0]
                    results[key]['r2'] = res[1]

best_mse = 100
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
print(len(results))
'''
test_model(X_train, X_test, y_test, 64, 50, 2, 32, 0.01)
'''
