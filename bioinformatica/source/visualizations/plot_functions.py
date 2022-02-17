import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances
from bioinformatica.source.preprocessing.correlation import *
from barplots import barplots
from pathlib import Path
import numpy as np


def balance_visualization(filename: str, plot_title: str, labels: list, counts: np.array):
    plt.figure(figsize=(4, 3))
    plt.grid(b=True, axis='both')
    plt.subplot(111)
    plt.bar(labels, counts)
    plt.title(plot_title)
    path = Path(__file__).parent
    plt.savefig(str(path) + '/dataset_balancing/' + filename, bbox_inches='tight')


def feature_correlations_visualization(filename: str, dataset: pd.DataFrame, features: list, labels: np.array, top: int,
                                       p_value: float):
    most_correlated = [[int(index) for index in score[0].split(' ')] for score in
                       sorted(features, key=lambda x: x[1], reverse=True)
                       if score[2] < p_value][:top]
    least_correlated = [[int(index) for index in score[0].split(' ')] for score in sorted(features, key=lambda x: x[1])
                        if score[2] < p_value][:top]

    mc_all_indices = list(set([x for y in [(x, y) for x, y in most_correlated] for x in y]))
    lc_all_indices = list(set([x for y in [(x, y) for x, y in least_correlated] for x in y]))

    sns.pairplot(pd.concat([dataset.iloc[:, mc_all_indices], pd.DataFrame(labels)], axis=1), hue=0)

    path = Path(__file__).parent
    plt.savefig(str(path) + '/features_correlations/' + 'MOST_' + filename, bbox_inches='tight')

    sns.pairplot(pd.concat([dataset.iloc[:, lc_all_indices], pd.DataFrame(labels)], axis=1), hue=0)

    plt.savefig(str(path) + '/features_correlations/' + 'least_' + filename, bbox_inches='tight')


def feature_distribution_visualization(filename: str, dataset: pd.DataFrame, labels: np.array, top_number: int):
    dist = euclidean_distances(dataset.T)
    most_distance_columns_indices = np.argsort(-np.mean(dist, axis=1).flatten())[:top_number]
    columns = dataset.columns[most_distance_columns_indices]

    fig, axes = plt.subplots(nrows=1, ncols=top_number, figsize=((top_number * 5), 5))

    for column, axis in zip(columns, axes.flatten()):
        head, tail = dataset[column].quantile([0.05, 0.95]).values.ravel()

        mask = ((dataset[column] < tail) & (dataset[column] > head)).values

        cleared_x = dataset[column][mask]
        cleared_y = labels.ravel()[mask]

        cleared_x[cleared_y == 0].hist(ax=axis, bins=20)
        cleared_x[cleared_y == 1].hist(ax=axis, bins=20)

        axis.set_title(column)
    fig.tight_layout()
    path = Path(__file__).parent
    plt.savefig(str(path) + '/features_distributions/' + filename)


def top_different_tuples_visualization(filename: str, dataset: pd.DataFrame, top_number: int):
    dist = euclidean_distances(dataset.T)
    dist = np.triu(dist)
    tuples = list(zip(*np.unravel_index(np.argsort(-dist.ravel()), dist.shape)))[:top_number]

    fig, axes = plt.subplots(nrows=1, ncols=top_number, figsize=((top_number * 5), 5))

    for (i, j), axis in zip(tuples, axes.flatten()):
        column_i = dataset.columns[i]
        column_j = dataset.columns[j]
        for column in (column_i, column_j):
            head, tail = dataset[column].quantile([0.05, 0.95]).values.ravel()
            mask = ((dataset[column] < tail) & (dataset[column] > head)).values
            dataset[column][mask].hist(ax=axis, bins=20, alpha=0.5)
        axis.set_title(f"{column_i} and {column_j}")
    fig.tight_layout()
    path = Path(__file__).parent

    plt.savefig(str(path) + '/top_different_tuples/' + filename)


def PCA_TSNE_visualization(filename: str, points: list, labels: list, algorithm: str):
    colors = np.array([
        "tab:blue",
        "tab:orange",
    ])

    path = Path(__file__).parent
    fig, ax = plt.subplots(nrows=1, ncols=1)
    xs, ys = [x[0] for x in points], [y[1] for y in points]
    ax.scatter(xs, ys, s=1, color=colors[labels])
    ax.xaxis.set_visible(True) #ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(True) #ax.yaxis.set_visible(False)
    if algorithm == 'PCA':
        plt.savefig(str(path) + '/decompositions/PCA' + filename)
    else:
        plt.savefig(str(path) + '/decompositions/TSNE' + filename)


def experiment_visualization(filename: str, results: pd.DataFrame):
    barplots(results,
             groupby=["model", "run_type"],
             show_legend=False,
             height=5,
             orientation="horizontal",
             path='barplots/' + filename + '{feature}.png',
             )
