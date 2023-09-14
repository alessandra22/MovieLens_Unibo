import pandas as pd


def load_csv(path, debug=False):
    print(f'Loading {path}...')
    df = pd.read_csv(path, delimiter=',')
    print(f'{path} loaded')

    if debug:
        print(f'Shape = {df.shape}')
        print(f'Index = {df.index}')
        print(f'Columns = {df.columns}')
        print(df.loc[12, :])
        print()

    return df


def get_data(debug=False):
    genome_scores_df = load_csv('./dataset/sources/genome-scores.csv', debug)
    genome_tags_df = load_csv('./dataset/sources/genome-tags.csv', debug)
    movies_df = load_csv('./dataset/sources/movies.csv', debug)
    ratings_df = load_csv('./dataset/sources/ratings.csv', debug)
    tags_df = load_csv('./dataset/sources/tags.csv', debug)
    # the relation between the tag given and the review left is not useful

    return genome_scores_df, genome_tags_df, movies_df, ratings_df, tags_df
