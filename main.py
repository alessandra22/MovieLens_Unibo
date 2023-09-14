from implementation.data_preprocessing import clean_dataframes
from dataset.data_acquisition import get_data, load_csv
from dataset.data_visualization import draw_graphs
from utils.utils import save_data_csv
from implementation.modeling.non_deep import start_algorithms
from implementation.modeling.deep_learning import deep_test
from implementation.modeling.tabular_learning import tabular_learning


# genome_scores_df, genome_tags_df, movies_df, ratings_df, tags_df = get_data()
# draw_graphs(genome_scores_df, movies_df, ratings_df)
# clean_df = clean_dataframes(genome_scores_df, genome_tags_df, movies_df, ratings_df, False)

# save_data_csv(clean_df, './clean_df.csv')
clean_df = load_csv('./clean_df.csv')
clean_df.set_index('movieId', inplace=True)

y_df = clean_df['rating']
X_df = clean_df.drop('rating', axis=1)

# start_algorithms(X_df, y_df)
# deep_test(X_df, y_df, 0.20, 0.15)
# deep_test(X_df, y_df, 0.20, 0.10)
# tabular_learning(X_df, y_df, 0.20, 0.15)
tabular_learning(X_df, y_df, 0.20, 0.10)
