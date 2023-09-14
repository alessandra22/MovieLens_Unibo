import pandas as pd
from numpy import arange
from matplotlib import pyplot as plt
import seaborn as sns
from utils.utils import get_genres_pool


def draw_mean(genome_scores_df):
    plt.figure(figsize=(8, 6), dpi=300)
    print('saving first plot (relevance)...')
    plt.clf()
    plt.cla()

    sns.displot(genome_scores_df['relevance'], bins=arange(0, 1.05, 0.05), kde=False)
    plt.xlabel('relevance')
    plt.ylabel('count')
    plt.title('Distribution of tags relevance in the dataset')
    plt.savefig("output/plots/tags.png", bbox_inches="tight")


def draw_ratings_total(ratings_df):
    plt.clf()
    plt.cla()

    print('saving second plot (total ratings)...')
    plt.figure(figsize=(8, 6), dpi=300)
    sns.displot(ratings_df['rating'], bins=arange(0, 5.5, 0.5), color='orange', kde=False)
    plt.xlabel('rating')
    plt.ylabel('count')
    plt.title('Distribution of ratings in the dataset')
    plt.savefig("output/plots/ratings.png", bbox_inches="tight")


def draw_ratings_mean(ratings_df):
    plt.clf()
    plt.cla()

    print('saving third plot (mean rating)...')
    plt.title('Distribution of mean ratings in the dataset')
    plt.figure(figsize=(8, 6), dpi=300)
    mean_serie = ratings_df.groupby('movieId')['rating'].mean()
    mean_serie = mean_serie.round(1)
    # plt.hist(mean_serie, bins=arange(0, 5.1, 0.1), color='red', alpha=0.7)
    # mean_serie.plot.kde(color='darkred')
    sns.displot(mean_serie, bins=arange(0, 5.1, 0.1), color='red', kde=True)
    plt.xlabel('mean rating')
    plt.ylabel('n. movies')
    plt.xticks(arange(0, 5.5, 0.5))
    plt.title('Distribution of mean ratings')
    plt.tick_params(axis='both', which='major', labelsize=5)
    plt.savefig("output/plots/mean_rating.png", bbox_inches="tight")


def draw_genres(movies_df):
    plt.clf()
    plt.cla()

    print('saving fourth plot (genres)...')
    plt.figure(figsize=(8, 6), dpi=300)
    genres = get_genres_pool(movies_df, False)
    genres_df = pd.DataFrame.from_dict({g: [genres.count(g)] for g in set(genres)}, orient='index')
    genres_df['count'] = genres_df[0]
    genres_df = genres_df.drop(columns=0)
    genres_df.plot(kind='bar', xlabel='Genre', ylabel='Count', color='green')
    plt.tick_params(axis='both', which='major', labelsize=7)
    # plt.legend(loc="upper right")
    plt.title('Distribution of genres in movies')
    plt.savefig("output/plots/genres.png", bbox_inches="tight")


def draw_graphs(genome_scores_df, movies_df, ratings_df):
    draw_mean(genome_scores_df)
    draw_ratings_total(ratings_df)
    draw_ratings_mean(ratings_df)
    draw_genres(movies_df)


def draw_loss(train_loss, val_loss, name):
    plt.cla()
    plt.clf()
    plt.plot(range(len(train_loss)), train_loss, label="Train", color='blue')
    plt.plot(range(len(val_loss)), val_loss, label="Validation", color='orange')
    plt.title(f"Loss variations for {name} test-val split")
    plt.legend(loc="best")
    plt.xlabel("Epoch")
    plt.ylabel("MSELoss")
    plt.savefig(f"output/net1{name}.png", bbox_inches="tight")
    plt.cla()
    plt.clf()
    plt.plot([i for i in range(20, 80)], train_loss[20:], label="Train", color='blue')
    plt.plot([i for i in range(20, 80)], val_loss[20:], label="Validation", color='orange')
    plt.title(f"Loss variations from epoch 20 for {name} test-val split")
    plt.legend(loc="best")
    plt.xlabel("Epoch")
    plt.xticks([i for i in range(20, 90, 10)])
    plt.ylabel("MSELoss")
    plt.savefig(f"output/net2{name}.png", bbox_inches="tight")

