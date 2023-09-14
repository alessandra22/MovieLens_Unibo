import time
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse, r2_score as r2

from implementation.data_preprocessing import pca_reduction


def print_metrics(model, y_test, y_pred):
    print(f'Metrics obtained for {model}:')
    print('\tMean squared error:', mse(y_test, y_pred))
    print('\tR-squared scores:', r2(y_test, y_pred))


def linear_regression(X_train, X_test, y_train, y_test):
    print('Fitting Linear Regression model...')
    lr_model = LinearRegression(n_jobs=-1)
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    print_metrics('Linear Regression', y_test, y_pred)


def naive_bayes(X_train, X_test, y_train, y_test, grid=False):
    if grid:
        print('Fitting Decision Tree model with grid parameters...')
        param_grid = {
            'alpha_1': [1e-7, 1e-6, 1e-5, 1e-4],
            'alpha_2': [1e-7, 1e-6, 1e-5, 1e-4],
            'lambda_1': [1e-7, 1e-6, 1e-5, 1e-4],
            'lambda_2': [1e-7, 1e-6, 1e-5, 1e-4]
        }
        start = time.time()
        grid_search = GridSearchCV(BayesianRidge(), param_grid, cv=5, n_jobs=-1, verbose=200)
        grid_search.fit(X_train, y_train)
        end = time.time()
        print('Time needed:', end - start, 'seconds')
        y_pred = grid_search.predict(X_test)
        print('Best parameters:', grid_search.best_params_)
    else:
        print('Fitting Naive Bayes model...')
        nb_model = BayesianRidge(alpha_1=1e-07, alpha_2=0.0001, lambda_1=0.0001, lambda_2=1e-07)
        nb_model.fit(X_train, y_train)
        y_pred = nb_model.predict(X_test)
    print_metrics('Naive Bayes', y_test, y_pred)


def decision_tree(X_train, X_test, y_train, y_test, grid=False):
    if grid:
        print('Fitting Decision Tree model with grid parameters...')
        param_grid = {
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 2, 3, 4],
            'max_features': [None, 'sqrt', 'log2'],
            'ccp_alpha': [0.0, 0.0020, 0.0030, 0.0035, 0.0040]
        }
        start = time.time()
        grid_search = GridSearchCV(DecisionTreeRegressor(random_state=22), param_grid, cv=5, n_jobs=-1, verbose=200)
        grid_search.fit(X_train, y_train)
        end = time.time()
        print('Time needed:', end - start, 'seconds')
        y_pred = grid_search.predict(X_test)
        print('Best parameters:', grid_search.best_params_)
    else:
        print('Fitting Decision Tree model...')
        dt_regr = DecisionTreeRegressor(ccp_alpha=0.0, criterion="poisson", max_depth=4, max_features=None, splitter="best", random_state=22)
        dt_regr.fit(X_train, y_train)
        y_pred = dt_regr.predict(X_test)
    print_metrics('Decision Tree', y_test, y_pred)


def random_forest(X_train, X_test, y_train, y_test, grid=False):
    if grid:
        print('Fitting Random Forest model with grid parameters...')
        param_grid = {
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            'n_estimators': [10, 20, 50, 100],
            'max_depth': [2, 3, 4],
            'max_features': ['sqrt', 'log2'],
            'ccp_alpha': [0.0030, 0.0035, 0.0040]
        }
        start = time.time()
        grid_search = GridSearchCV(RandomForestRegressor(random_state=22), param_grid, cv=5, n_jobs=-1, verbose=200)
        grid_search.fit(X_train, y_train)
        end = time.time()
        print('Time needed:', end - start, 'seconds')
        y_pred = grid_search.predict(X_test)
        print('Best parameters:', grid_search.best_params_)
    else:
        print('Fitting Random Forest model...')
        rf_regr = RandomForestRegressor(random_state=22, n_jobs=-1)
        rf_regr.fit(X_train, y_train)
        y_pred = rf_regr.predict(X_test)
    print_metrics('Random Forest', y_test, y_pred)


def support_vector_regressor(X_train, X_test, y_train, y_test, grid=False):
    if grid:
        print('Fitting Support Vector Machine model with grid parameters...')
        param_grid = {
            'kernel': ['rbf', 'linear', 'poly'],
            'C': [0.001, 0.01, 0.1, 1, 10],
            'gamma': ['auto', 'scale'],
            'degree': [3, 4, 5]
        }
        start = time.time()
        grid_search = GridSearchCV(SVR(), param_grid, cv=5, n_jobs=-1, verbose=200)
        grid_search.fit(X_train, y_train)
        end = time.time()
        print('Time needed:', end - start, 'seconds')
        y_pred = grid_search.predict(X_test)
        print('Best parameters:', grid_search.best_params_)
    else:
        print('Fitting Support Vector Machine model...')
        svr_model = SVR(C=10, degree=3, gamma="auto", kernel="rbf")
        svr_model.fit(X_train, y_train)
        y_pred = svr_model.predict(X_test)
    print_metrics('Support Vector Machine', y_test, y_pred)


def k_nearest_neighbors(X_train, X_test, y_train, y_test, grid=False):
    if grid:
        print('Fitting K nearest neighbors model with grid parameters...')
        param_grid = {
            'n_neighbors': [1, 2, 3, 5, 7, 10, 15],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [1, 5, 10, 20, 30, 40],
            'p': [1, 2]
        }
        start = time.time()
        grid_search = GridSearchCV(KNR(), param_grid, cv=5, n_jobs=-1, verbose=200)
        grid_search.fit(X_train, y_train)
        end = time.time()
        print('Time needed:', end - start, 'seconds')
        y_pred = grid_search.predict(X_test)
        print('Best parameters:', grid_search.best_params_)
    else:
        print('Fitting K nearest neighbors model...')
        knn_model = KNR(n_neighbors=10, weights="distance", algorithm="auto", leaf_size=1, p=2)
        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(X_test)
    print_metrics('K nearest neighbors', y_test, y_pred)


def start_algorithms(X_df, y_df, grid=False):
    X_train, X_test, y_train, y_test = pca_reduction(X_df, y_df)
    print('Dataset successfully reduced')
    linear_regression(X_train, X_test, y_train, y_test)
    naive_bayes(X_train, X_test, y_train, y_test, grid)
    decision_tree(X_train, X_test, y_train, y_test, grid)
    random_forest(X_train, X_test, y_train, y_test, grid)
    support_vector_regressor(X_train, X_test, y_train, y_test, grid)
    k_nearest_neighbors(X_train, X_test, y_train, y_test, grid)
