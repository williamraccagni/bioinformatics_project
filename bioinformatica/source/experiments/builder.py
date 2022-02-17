from bioinformatica.source.experiments.utils import *
from bioinformatica.source.datasets.loader import get_holdouts
from bioinformatica.source.preprocessing.pipeline import pipeline
from bioinformatica.source.models.builder import Model
from bioinformatica.source.experiments.evaluation import test_models
from bioinformatica.source.preprocessing.elaboration import balance
from bioinformatica.source.type_hints import *
from bioinformatica.source.datasets.loader import get_data
import pandas as pd
from pathlib import Path

#tsdsds

import numpy as np


'''
Experiment class
Class used to perform experiments
An experiment is intended as a cell line to be analyzed, performing data retrieval, data preprocessing and machine learning
algorithms on it.
An Experiment object takes several arguments:
- experiment_id: used to identify an experiment, and to collect results from different experiments
- data_parameters: a tuple of tuple, to indicate cell line, window size and if the data must be enhancers or promoters and 
a string inside the external tuple to indicate if it is epigenomic or sequences dataset
- holdout_parameters: a tuple to indicate the number of split for holdouts, test set size and random state
- alphas: values to use with statistical tests, like wilcoxon
- defined_algorithms: algorithms and models defined for the experiment, into definition.py file
- balance_type: default is None, can be set to 'under_sample', 'over_sample' or SMOTE. Sequence data
- save_result: if true, will save results of the experiment in a csv file, ready to be visualized using visualization.py
    function
- dataset_row_reduction: a parameter to take first n rows of a dataset to train
- execute_pipeline: if data preprocessing is needed

methods:
- execute(): runs the experiment. Retrieves the data, create holdouts (for train and test), builds the models, execute trainings
    and test the trained models
    
- evaluate(): executes statistical tests, ordered by metric and by statistical test 

Example of use:
    experiment_id = 1
    data_type = 'epigenomic'
    cell_line, window_size, epigenomic_type = 'K562', 200, 'enhancers'
    n_split, test_size, random_state = 1, 0.2, 1
    balance = 'under_sample'
    save_results = False
    dataset_row_reduction = None
    execute_pipeline = True
    defined_algorithms = define_models()
    holdout_parameters = (n_split, test_size, random_state)
    data_parameters = ((cell_line, window_size, epigenomic_type), data_type)
    alphas = [0.05]
    experiment = Experiment(experiment_id, data_parameters, holdout_parameters, alphas, defined_algorithms, balance, 
                            save_results, dataset_row_reduction, execute_pipeline)
    experiment.execute()
    experiment.evaluate()
    experiment.print_model_info('all')
'''


class Experiment:
    def __init__(self, experiment_id: int, data_parameters: Tuple[Tuple[str, int, str], str], holdout_parameters:
            Tuple[int, float, int], alphas: List[float], defined_algorithms: Dict[str, List], balance_type: str = None,
            save_results: bool = False, dataset_row_reduction: int = None, execute_pipeline: bool = True):
        if save_results:
            self.__save_results = True
        else:
            self.__save_results = False
        self.__experiment_id = experiment_id
        self.__data_type = data_parameters[1]
        self.__data_parameters, self.__holdout_parameters = data_parameters, holdout_parameters
        self.__models = []
        self.__alphas = alphas
        self.__statistical_tests_scores = {}
        self.__balance_type = balance_type
        self.__defined_algorithms = defined_algorithms
        self.__dataset_row_reduction = dataset_row_reduction
        self.__execute_pipeline = execute_pipeline

    def execute(self):
        if self.__execute_pipeline:
            dataset, labels = pipeline((self.__data_parameters, self.__holdout_parameters[-1]))
        else:
            #dataset, labels = get_data(self.__data_parameters)

            #EPIGENOMIC

            print("andata bene")

            cell_line = 'GM12878'
            epigenomic_type = 'promoters' #'enhancers'

            path = Path(__file__).parent
            dataset = pd.read_csv(
                str(path) + '/preprocessati/' + cell_line + '_' + epigenomic_type + '_preprocessed.csv')
            dataset = dataset.drop(dataset.columns[0], axis=1)
            labels = np.genfromtxt(
                str(path) + '/preprocessati/' + cell_line + '_' + epigenomic_type + '_labels.csv')
            labels = np.array([int(i) for i in labels])




            #SEQUENCES

            # cell_line = 'GM12878'
            # epigenomic_type = 'enhancers'
            # # epigenomic_type = 'promoters'
            #
            # path = Path(__file__).parent
            # dataset = pd.read_csv(str(path) + '/sequences/' + cell_line + '_' + epigenomic_type + '_sequences.csv')
            # dataset = dataset.drop(dataset.columns[0], axis=1)
            # dataset = dataset.astype(int)
            # labels = np.genfromtxt(str(path) + '/sequences/' + cell_line + '_' + epigenomic_type + '_labels.txt')
            # labels = np.array([int(i) for i in labels])
            #
            # # print(dataset)
            # # print('------------------')
            # # print('num di labels: ', len(labels))
            # # print('\n', labels)
            #
            # # exit()




        if self.__dataset_row_reduction:
            if self.__data_type == 'epigenomic':
                dataset = dataset.head(self.__dataset_row_reduction)
                labels = labels[:self.__dataset_row_reduction]
            else:
                dataset = dataset[:self.__dataset_row_reduction]
                labels = labels[:self.__dataset_row_reduction]
        for algorithm in self.__defined_algorithms:
            for name, hyperparameters in self.__defined_algorithms.get(algorithm):
                model = Model(algorithm, name, False if type(hyperparameters) == dict else True)
                model.build(hyperparameters)
                self.__models.append(model)
        for holdout, data in enumerate(get_holdouts(dataset, labels, self.__holdout_parameters, self.__data_type)):
            training_data, test_data = data
            X_train, y_train = training_data
            X_test, y_test = test_data
            if self.__balance_type:
                resampled_X_train, resampled_y_train = balance(X_train, y_train, self.__holdout_parameters[-1],
                                                               self.__balance_type, self.__data_type)
                training_data = (resampled_X_train, resampled_y_train)
            for model in self.__models:
                model.train(training_data)
                y_train_prediction = model.predict(X_train)
                y_test_prediction = model.predict(X_test)
                for metric in metrics:
                    model.test_metrics(metric, (y_train, y_train_prediction))
                for metric in metrics:
                    model.test_metrics(metric, (y_test, y_test_prediction))
        if self.__save_results:
            self.__results_to_dataframe()

    def evaluate(self):
        for alpha in self.__alphas:
            for statistical_test in statistical_tests:
                self.__statistical_tests_scores[statistical_test.__name__] = []
                for metric in metrics:
                    self.__statistical_tests_scores.get(statistical_test.__name__) \
                        .append(test_models(self.__models, statistical_test, metric, alpha))

    def get_models(self) -> List[Model]:
        return self.__models

    def get_best_models(self) -> Dict[str, List[Tuple[Model, float, str, float]]]:
        return self.__statistical_tests_scores

    def print_model_info(self, models: str = 'models'):
        if models == 'models':
            for model in self.__models:
                print_model(model)
        elif models == 'best':
            for statistical_test in statistical_tests:
                for score in self.__statistical_tests_scores.get(statistical_test.__name__):
                    pprint((score[0].get_name(), score[1:]))
        elif models == 'all':
            for model in self.__models:
                print_model(model)
            print('---- Best models ----')
            for statistical_test in statistical_tests:
                for score in self.__statistical_tests_scores.get(statistical_test.__name__):
                    pprint((score[0].get_name(), score[1:]))

    def __results_to_dataframe(self):
        columns = ['model', 'run_type', 'accuracy', 'average precision', 'balanced accuracy', 'AUROC']
        score_values = [[]] * len(metrics)
        for model in self.__models:
            model_scores = list(model.get_scores().values())
            for i, score in enumerate(model_scores):
                score_values[i] = score_values[i] + score
        models_names = [[model.get_name()] * 2 * self.__holdout_parameters[0] for model in self.__models]
        run_type_values = ['train', 'test'] * len(self.__models) * self.__holdout_parameters[0]
        models_names = [x for y in models_names for x in y]
        values = {columns[0]: models_names, columns[1]: run_type_values}
        for i, column in enumerate(columns[2:]):
            values[column] = score_values[i]

        results = pd.DataFrame(values)
        path = Path(__file__).parent
        results.to_csv(str(path) + '/results/experiment_results_' + str(self.__experiment_id) + '.csv')




        # # flavio
        # results_to_be_tested = results[results['run_type'] == 'test']
        # results_to_be_tested.to_csv(str(path) + '/results_to_be_tested/experiment_results_' + str(self.__experiment_id) + '.csv')


