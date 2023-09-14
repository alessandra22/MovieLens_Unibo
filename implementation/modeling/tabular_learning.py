from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse, r2_score as r2
from utils.utils import save_data_pkl
import matplotlib.pyplot as plt
import torch


def plot_importance(X_df, results_tabular, vs):
    BEST = 25

    i = 0
    important_features = dict()
    best_features = dict()
    for c in X_df.columns:
        if results_tabular[i] != 0.0:
            important_features[c] = results_tabular[i]
        i += 1

    best = [important_features[k] for k in important_features.keys()]
    best.sort()
    best = best[-BEST:]

    for k in important_features.keys():
        if important_features[k] in best:
            best_features[k] = important_features[k]

    plt.figure(figsize=(20, 20), dpi=300)
    plt.barh(range(len(best_features)), best_features.values(), align='center')
    plt.yticks(range(len(best_features)), best_features.keys())
    plt.tick_params(axis='both', which='major', labelsize=30)
    if vs == 15:
        plt.savefig("output/tabnet_best20_65-15-20.png", bbox_inches="tight")
    else:
        plt.savefig("output/tabnet_best20_70-10-20.png", bbox_inches="tight")


def tabular_learning(X_df, y_df, ts, vs):
    if vs == 0.15:
        PATH = 'models-65-15-20/'
    else:
        PATH = 'models-70-10-20/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=ts, random_state=22)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=vs, random_state=22)

    model = TabNetRegressor()
    model.fit(
        X_train.values.astype(float),
        y_train.values.reshape(-1, 1).astype(float),
        eval_set=[(X_train.values.astype(float), y_train.values.reshape(-1, 1).astype(float)),
                  (X_val.values.astype(float), y_val.values.reshape(-1, 1).astype(float))],
        eval_name=['train', 'validation']
    )
    y_pred = model.predict(X_test.values.astype(float))
    model.save_model(PATH + 'TabNet')
    save_data_pkl(model.feature_importances_, PATH + 'Tabnet_feat.pkl')

    print(f'Metrics obtained for tabular learning:')
    print('\tMean squared error:', mse(y_test.values, y_pred))
    print('\tR-squared scores:', r2(y_test.values, y_pred))

    plot_importance(X_df, model.feature_importances_, vs)
