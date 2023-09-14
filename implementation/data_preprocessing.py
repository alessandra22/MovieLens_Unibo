from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils.utils import get_genres_pool
from sklearn.decomposition import PCA
import numpy as np


def intersect_dataset(genome_scores_df, genome_tags_df, movies_df, ratings_df, debug=False):
    def get_intersection():
        movies_in_gs = set(genome_scores_df.loc[:, 'movieId'])
        movies_in_movies = set(movies_df.loc[:, 'movieId'])
        movies_in_ratings = set(ratings_df.loc[:, 'movieId'])
        intersection = movies_in_gs.intersection(movies_in_movies, movies_in_ratings)

        if debug:
            print('movies in genome-scores.csv =', len(movies_in_gs))
            print('movies in movies.csv =', len(movies_in_movies))
            print('movies in ratings.csv =', len(movies_in_ratings))
            print('movies in intersection =', len(intersection))

        return intersection

    old_shape_gs = genome_scores_df.shape
    old_shape_movies = movies_df.shape
    old_shape_ratings = ratings_df.shape

    movies_in_all_dataframes = get_intersection()
    genome_scores_df = genome_scores_df[genome_scores_df['movieId'].isin(movies_in_all_dataframes)]
    movies_df = movies_df[movies_df['movieId'].isin(movies_in_all_dataframes)]
    ratings_df = ratings_df[ratings_df['movieId'].isin(movies_in_all_dataframes)]

    if debug:
        print(old_shape_gs, genome_scores_df.shape, old_shape_gs[0] - genome_scores_df.shape[0])
        print(old_shape_movies, movies_df.shape, old_shape_movies[0] - movies_df.shape[0])
        print(old_shape_ratings, ratings_df.shape, old_shape_ratings[0] - ratings_df.shape[0])

    return genome_scores_df, genome_tags_df, movies_df, ratings_df


def drop_nan(genome_scores_df, genome_tags_df, movies_df, ratings_df):
    dim = genome_scores_df.shape[0]
    genome_scores_df = genome_scores_df.dropna()
    print(f'{dim - genome_scores_df.shape[0]} rows dropped for genome_scores_df.')

    dim = genome_tags_df.shape[0]
    genome_tags_df = genome_tags_df.dropna()
    print(f'{dim - genome_tags_df.shape[0]} rows dropped for genome_tags_df.')

    dim = movies_df.shape[0]
    movies_df = movies_df.dropna()
    print(f'{dim - movies_df.shape[0]} rows dropped for movies_df.')

    dim = ratings_df.shape[0]
    ratings_df = ratings_df.dropna()
    print(f'{dim - ratings_df.shape[0]} rows dropped for ratings_df.')

    return genome_scores_df, genome_tags_df, movies_df, ratings_df


def clean_dataframes(genome_scores_df, genome_tags_df, movies_df, ratings_df, debug=False):
    def add_genres(pool_genres, joined_df):
        i = 0
        for genre in pool_genres:
            i += 1
            if debug:
                print(f'Converting {genre}, {i}/20')
            joined_df[genre] = np.where(joined_df['genres'].str.contains(genre), True, False)

        if debug:
            print(joined_df.columns)
            print(joined_df.iloc[78])

        return joined_df

    genome_scores_df, genome_tags_df, movies_df, ratings_df = drop_nan(genome_scores_df, genome_tags_df, movies_df,
                                                                       ratings_df)
    genome_scores_df, genome_tags_df, movies_df, ratings_df = intersect_dataset(genome_scores_df, genome_tags_df,
                                                                                movies_df, ratings_df, debug)
    mean_serie = ratings_df.groupby('movieId')['rating'].mean()

    joined_dataset = genome_scores_df.merge(genome_tags_df, on='tagId')
    joined_dataset = joined_dataset.drop('tagId', axis=1)
    joined_dataset = joined_dataset.pivot_table(index='movieId', columns='tag', values='relevance')
    joined_dataset = joined_dataset.merge(movies_df, on='movieId')
    joined_dataset = joined_dataset.merge(mean_serie, on='movieId')  # kept instead of single evaluations (useless)

    joined_dataset = add_genres(get_genres_pool(movies_df, True, debug), joined_dataset)

    joined_dataset = joined_dataset.drop('title', axis=1)  # not relevant
    joined_dataset = joined_dataset.drop('genres', axis=1)  # individual genre columns kept instead of single string
    joined_dataset['No-Genre'] = joined_dataset['(no genres listed)']  # keeping naming scheme
    joined_dataset = joined_dataset.drop('(no genres listed)', axis=1)

    joined_dataset.set_index('movieId', inplace=True)  # movie ID as index, so it won't be calculated while searching

    if debug:
        print(joined_dataset)
        print(joined_dataset.loc[206499, :])
        print(joined_dataset.iloc[72, :])

    return joined_dataset


def dataframe_scaler(X_df, y_df):
    print('Splitting dataframes')
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.30, random_state=22, shuffle=True)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    print('Scaling dataframes')
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test


def pca_reduction(X_df, y_df):
    X_train, X_test, y_train, y_test = dataframe_scaler(X_df, y_df)
    pca = PCA(0.9)
    print('Fitting PCA')
    pca.fit(X_train)
    print('Transforming with PCA')
    return pca.transform(X_train), pca.transform(X_test), y_train, y_test


def get_validation_split(X_df, y_df, ts, vs):
    print('Splitting the dataframe into train, validation and test set')
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, random_state=22, test_size=ts, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=22, test_size=vs)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    print('Scaling')
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    pca = PCA(0.9)
    print('Fitting PCA')
    pca.fit(X_train)
    print('Transforming with PCA')
    X_train = pca.transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test
