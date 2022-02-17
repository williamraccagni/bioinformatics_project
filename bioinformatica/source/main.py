from scipy.stats import wilcoxon

from bioinformatica.source.experiments.builder import Experiment
from bioinformatica.source.experiments.definition import *
from bioinformatica.source.preprocessing.elaboration import drop_constant_features, robust_zscoring
from bioinformatica.source.preprocessing.feature_selection import boruta
from bioinformatica.source.visualizations.visualization import *
from bioinformatica.source.datasets.loader import get_data

from bioinformatica.source.preprocessing.imputation import imputation, nan_filter

'''
Example of an experiment setup:
    experiment_id = 1
    dataset_type = 'epigenomic'
    cell_line, window_size, epigenomic_type = 'K562', 200, 'enhancers'
    n_split, test_size, random_state = 1, 0.2, 1
    balance = 'under_sample'
    save_results = False
    dataset_row_reduction = None
    execute_pipeline = True
    defined_algorithms = define_models()
    holdout_parameters = (n_split, test_size, random_state)
    data_parameters = ((cell_line, window_size, epigenomic_type), dataset_type)
    alphas = [0.05]
    experiment = Experiment(experiment_id, data_parameters, holdout_parameters, alphas, defined_algorithms, balance, 
                            save_results, dataset_row_reduction, execute_pipeline)
    experiment.execute()
    experiment.evaluate()
    experiment.print_model_info('all')

'''

'''
Visualization allows to plot many kind of data. Options include PCA visualization, the balancing of the dataset, the top 
    n different tuples, feature correlation, feature distribution and TSNE.
    Experiment results images can be found inside source.barplots folder
    Below you can find how to execute any visualization 

Example of visualization setup:
    
    dataset_type = 'epigenomic'
    cell_line, window_size, epigenomic_type = 'K562', 200, 'enhancers'
    n_split, test_size, random_state = 1, 0.2, 1
    data_parameters = ((cell_line, window_size, epigenomic_type), dataset_type)
    dataset, labels = get_data(data_parameters)
    dataset = imputation(dataset)
    
    make_visualization('experiment_results')
    make_visualization('PCA', dataset, labels, cell_line, epigenomic_type, dataset_type, PCA_n_components=50)
    make_visualization('balancing', dataset, labels, cell_line, epigenomic_type, dataset_type)
    make_visualization('top_different_tuples', dataset, labels, cell_line, epigenomic_type, dataset_type,
                       top_different_tuples=5)
    make_visualization('feature_correlations', dataset, labels, cell_line, epigenomic_type, dataset_type,
                       top_features_for_correlation=3, p_value_correlation=0.05)
    make_visualization('feature_distribution', dataset, labels, cell_line, epigenomic_type, dataset_type,
                       top_feature_distribution=5)
    make_visualization('TSNE', dataset, labels, cell_line, epigenomic_type, dataset_type,
                       TSNE_n_components=2, PCA_before_TSNE=True, PCA_n_components=75, TSNE_perplexity=40)
'''


if __name__ == '__main__':
    dataset_type = 'epigenomic'
    cell_line, window_size, epigenomic_type = 'GM12878', 200, 'promoters'#'enhancers'
    n_split, test_size, random_state = 10, 0.2, 1
    data_parameters = ((cell_line, window_size, epigenomic_type), dataset_type)
    dataset, labels = get_data(data_parameters)

    dataset, labels = nan_filter(dataset, labels)
    dataset = imputation(dataset)

    dataset = drop_constant_features(dataset)
    dataset = robust_zscoring(dataset)

    dataset = filter_uncorrelated(dataset, labels, 0.01, 0.05)
    dataset = filter_correlated_features(dataset, 0.01, 0.95)

    make_visualization('feature_correlations', dataset, labels, cell_line, epigenomic_type, dataset_type,
                       top_features_for_correlation=3, p_value_correlation=0.05)








    # dataset_type = 'sequences' #'epigenomic'
    # cell_line, window_size, epigenomic_type = 'GM12878', 200, 'promoters'
    # n_split, test_size, random_state = 10, 0.2, 1
    # data_parameters = ((cell_line, window_size, epigenomic_type), dataset_type)
    # dataset, labels = get_data(data_parameters)

    # #Class Balance
    #
    # dataset, labels = nan_filter(dataset, labels)
    # dataset = imputation(dataset)
    # # make_visualization('balancing', dataset, labels, cell_line, epigenomic_type, dataset_type)
    #
    # #Correlation
    #
    # dataset = drop_constant_features(dataset)
    # dataset = robust_zscoring(dataset)
    #
    # dataset = filter_uncorrelated(dataset, labels, 0.01, 0.05) #tra fetures e etichette
    #
    # make_visualization('feature_correlations', dataset, labels, cell_line, epigenomic_type, dataset_type,
    #                    top_features_for_correlation=3, p_value_correlation=0.01)
    # make_visualization('feature_distribution', dataset, labels, cell_line, epigenomic_type, dataset_type,
    #                    top_feature_distribution=5)
    # make_visualization('top_different_tuples', dataset, labels, cell_line, epigenomic_type, dataset_type,
    #                    top_different_tuples=5)
    # dataset = filter_correlated_features(dataset, 0.01, 0.95)  # solo features
    #
    # #Boruta
    #
    # boruta(dataset, labels, 200, 0.05, random_state)
    #
    # path = Path(__file__).parent
    # dataset.to_csv(str(path) + '/preprocessati/' + cell_line + '_' + epigenomic_type + '_preprocessed.csv')
    # np.savetxt(str(path) + '/preprocessati/' + cell_line + '_' + epigenomic_type + '_labels.csv', labels)

    # make_visualization('PCA', dataset, labels, cell_line, epigenomic_type, dataset_type, PCA_n_components=50)
    # make_visualization('TSNE', dataset, labels, cell_line, epigenomic_type, dataset_type,
    #                    TSNE_n_components=2, PCA_before_TSNE=True, PCA_n_components=75, TSNE_perplexity=40)



    # make_visualization('experiment_results')
    # make_visualization('PCA', dataset, labels, cell_line, epigenomic_type, dataset_type, PCA_n_components=50)
    # make_visualization('balancing', dataset, labels, cell_line, epigenomic_type, dataset_type)
    # make_visualization('top_different_tuples', dataset, labels, cell_line, epigenomic_type, dataset_type,
    #                    top_different_tuples=5)
    # make_visualization('feature_correlations', dataset, labels, cell_line, epigenomic_type, dataset_type,
    #                    top_features_for_correlation=3, p_value_correlation=0.05)
    # make_visualization('feature_distribution', dataset, labels, cell_line, epigenomic_type, dataset_type,
    #                    top_feature_distribution=5)
    # make_visualization('TSNE', dataset, labels, cell_line, epigenomic_type, dataset_type,
    #                    TSNE_n_components=2, PCA_before_TSNE=True, PCA_n_components=75, TSNE_perplexity=40)










    # experiment_id = 52
    # dataset_type = 'epigenomic'
    # cell_line, window_size, epigenomic_type = 'GM12878', 200, 'promoters' #'enhancers'
    # n_split, test_size, random_state = 10, 0.2, 1
    # balance = None #'under_sample' #SMOTE
    # save_results = True
    # dataset_row_reduction = None
    # execute_pipeline = False #True
    # defined_algorithms = define_models()
    # holdout_parameters = (n_split, test_size, random_state)
    # data_parameters = ((cell_line, window_size, epigenomic_type), dataset_type)
    # alphas = [0.05]
    # experiment = Experiment(experiment_id, data_parameters, holdout_parameters, alphas, defined_algorithms, balance,
    #                         save_results, dataset_row_reduction, execute_pipeline)
    # experiment.execute()
    # experiment.evaluate()
    # experiment.print_model_info('all')
    #
    # make_visualization('experiment_results')



























































    # experiment_id = 31
    #
    # #flavio
    # path = Path(__file__).parent
    # #dai al file questo nome
    # wilcoxon_df = pd.read_csv(str(path) + '/experiments/results_to_be_tested/experiment_results_' + str(experiment_id) + '.csv')
    # models = list(wilcoxon_df['model'].unique())
    # metrics = ['accuracy', 'average precision', 'balanced accuracy', 'AUROC']
    # best = {metric: None for metric in metrics}
    # for metric in metrics:
    #     by_metric_df = wilcoxon_df[metric]
    #     best_model, best_score = models[0], by_metric_df.head(10).values
    #     for model in models:
    #         comparison_scores = by_metric_df.loc[wilcoxon_df['model'] == model].values
    #         if model == best_model:
    #             continue
    #         stats, p_value = wilcoxon(best_score, comparison_scores)
    #         if p_value < alphas[0]:
    #             if comparison_scores.mean() > best_score.mean():
    #                 best_model, best_score = model, comparison_scores
    #     best[metric] = (model, best_score.mean())
    #
    # with open(str(path) + '/experiments/results_to_be_tested/best_models_' + str(experiment_id) + '.txt', 'w') as f:
    #     for k, v in best.items():
    #         dictionary_content = "best model for " + k + " is: " + str(v[0]) + " with a mean score of " + str(v[1]) + "\n"
    #         f.write(dictionary_content)















































    # # # MEDIE !!!!!
    # # cell_line = ['GM12878','HEK293','K562']
    # # region_type = ['enhancers', 'promoters']
    # # metrics = ['accuracy','average precision','balanced accuracy', 'AUROC']
    # #
    # # for cell in cell_line:
    # #     for region in region_type:
    # #         dataset = pd.read_csv('/home/willy/Desktop/avg_ris/' + cell + '_' + region + '/experiment_results_' + region + '.csv')
    # #         models = list(dataset['model'].unique())
    # #         means = {model: None for model in models}
    # #         for model in models:
    # #             medie = dataset[dataset['model'] == model].mean()
    # #             means[model] = {metrics[i]: medie[i] for i in range(4)}
    # #         df = pd.DataFrame(means).T
    # #         df.to_csv('/home/willy/Desktop/avg_ris/' + cell + '_' + region + '/means_' + region + '.csv')
    # #
    # #
    # #
    # #
    # #         #dataset = pd.read_csv('/home/federico/PycharmProjects/bioinformatica/bioinformatica/source/experiments/'
    # #         #                      'results2/experiment_results_8.csv')
    # #         #for y in dataset.groupby(['model']):
    # #             # x = y[1]
    # #             # print(x['model'].tolist()[0], x['accuracy'].mean(), x['average precision'].mean(),
    # #             #       x['balanced accuracy'].mean(), x['AUROC'].mean(), '\n\n')
    # #
    #
    #
    #
    # # STRONZO
    # cell_line = 'GM12878'
    # # epigenomic_type = 'enhancers'
    # epigenomic_type = 'promoters'
    #
    # dataset_type = 'sequences'
    #
    # path = Path(__file__).parent
    # dataset = pd.read_csv(str(path) + '/sequences/' + cell_line + '_' + epigenomic_type + '_sequences.csv')
    # dataset = dataset.drop(dataset.columns[0], axis=1)
    # dataset = dataset.astype(int)
    # labels = np.genfromtxt(str(path) + '/sequences/' + cell_line + '_' + epigenomic_type + '_labels.txt')
    # labels = np.array([int(i) for i in labels])
    #
    # print(dataset)
    # print(labels)
    #
    #
    # # make_visualization('PCA', dataset, labels, cell_line, epigenomic_type, dataset_type, PCA_n_components=50)
    # #make_visualization('TSNE', dataset, labels, cell_line, epigenomic_type, dataset_type,
    #          #          TSNE_n_components=2, PCA_before_TSNE=True, PCA_n_components=75, TSNE_perplexity=40)