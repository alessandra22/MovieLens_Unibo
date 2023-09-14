import pandas as pd
import pickle


def get_genres_pool(df, need_set, debug=False):
    def add_film(x):
        for g in x.split(sep='|'):
            if need_set:
                pool_film.add(g)
            else:
                pool_film.append(g)

    if need_set:
        pool_film = set()
    else:
        pool_film = list()
    df['genres'].apply(add_film)
    if debug:
        print(pool_film)
        print(len(pool_film))

    return pool_film


def save_data_csv(df, path):
    print(f'Saving data on {path}')
    df.to_csv(path)
    print(f'{path} saved')


def load_data_csv(path):
    print(f'Loading {path}')
    df = pd.read_csv(path)
    print(f'{path} loaded')
    return df


def save_data_pkl(my_data, path):
    pickle.dump(my_data, open(path, 'wb'))


def load_data_pkl(path):
    return pickle.load(open(path, 'rb'))

