from bioinformatica.source.preprocessing.decompositions import PCA_function, TSNE_function
from bioinformatica.source.visualizations.plot_functions import *
import os
from pathlib import Path


def make_visualization(visualization_type: str, dataset: pd.DataFrame or np.array = None, labels: np.array = None,
                       cell_line: str = None, epigenomic_type: str = None, dataset_type: str = None,
                       top_feature_distribution: int = None, top_features_for_correlation: int = None,
                       top_different_tuples: int = 5, p_value_correlation: float = None, PCA_before_TSNE: bool = True,
                       PCA_n_components: int = 50, TSNE_n_components: int = 2, TSNE_perplexity: int = 30,
                       random_state: int = 1):
    if visualization_type == 'experiment_results':
        path = Path(__file__).parent.parent
        experiment_files = os.listdir(str(path) + '/experiments/results')
        if len(experiment_files) > 0:#1:
            #experiment_files = experiment_files[1:]
            for file in experiment_files:
                dataframe = pd.read_csv(str(path) + '/experiments/results/' + file)
                dataframe.drop(dataframe.columns[0], inplace=True, axis=1)
                experiment_visualization(str(file)[:-4] + '_', dataframe)
        else:
            print('No experiment has been made, so any result can\'t be plotted')
    elif visualization_type == 'PCA':
        dataset = PCA_function(dataset, PCA_n_components, random_state)
        PCA_TSNE_visualization(cell_line + '_' + epigenomic_type + '_' + dataset_type, dataset, labels, 'PCA')
    elif visualization_type == 'TSNE':
        dataset = TSNE_function(dataset, TSNE_n_components, random_state, TSNE_perplexity, PCA_before_TSNE,
                                PCA_n_components)
        #quiprimo
        print(dataset)
        dataset.to_csv('/home/willy/Desktop/STRONZO.csv')

        #PCA_TSNE_visualization(cell_line + '_' + epigenomic_type + '_' + dataset_type, dataset, labels, 'TSNE')
    elif visualization_type == 'balancing':
        ones = np.count_nonzero(labels == 1)
        zeros = np.count_nonzero(labels == 0)
        balance_visualization(cell_line + '_' + dataset_type + '_' + epigenomic_type + '_balance.png', cell_line + ', '
                              + epigenomic_type + ' class balance', ['0', '1'], [zeros, ones])
    elif visualization_type == 'top_different_tuples':
        top_different_tuples_visualization(cell_line + '_' + dataset_type + '_' + epigenomic_type +
                                           '_different_tuples.png', dataset, top_different_tuples)
    elif visualization_type == 'feature_correlations':
        features = feature_correlation(dataset)
        feature_correlations_visualization(cell_line + '_' + dataset_type + '_' + epigenomic_type + '_correlation.png',
                                           dataset, features, labels, top_features_for_correlation, p_value_correlation)
    elif visualization_type == 'feature_distribution':
        feature_distribution_visualization(cell_line + '_' + dataset_type + '_' + epigenomic_type +
                                           '_feature_distribution.png', dataset, labels, top_feature_distribution)
