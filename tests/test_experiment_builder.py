from bioinformatica.source.experiments.builder import Experiment
from bioinformatica.source.experiments.utils import metrics, statistical_tests
from tests.dummy_models import define_models


def test_experiment():
    experiment_id = 1
    data_type = 'epigenomic'
    cell_line, window_size, epigenomic_type = 'K562', 200, 'enhancers'
    n_split, test_size, random_state = 1, 0.2, 1
    balance = 'under_sample'
    save_results = False
    dataset_row_reduction = 1000
    execute_pipeline = False
    defined_algorithms = define_models()
    holdout_parameters = (n_split, test_size, random_state)
    data_parameters = ((cell_line, window_size, epigenomic_type), data_type)
    alphas = [0.05]
    experiment = Experiment(experiment_id, data_parameters, holdout_parameters, alphas, defined_algorithms, balance,
                            save_results, dataset_row_reduction, execute_pipeline)

    experiment.execute()
    assert len(experiment.get_models()) == len(defined_algorithms.values()), 'not all models were built'
    total_scores = 0
    for model in experiment.get_models():
        total_scores += len([value for scores in model.get_scores().values() for value in scores])
    assert total_scores == n_split * len(metrics) * 2 * len(experiment.get_models()), \
        'not all models were correctly tested'

    experiment.evaluate()
    assert len(experiment.get_best_models().items()) * len(metrics) == len(statistical_tests) * len(metrics) * len(alphas), \
        'not all statistical tests were executed correctly or some alpha value may not have been tested'

    try:
        experiment.get_models()
    except:
        'error in models retrieving'

    try:
        experiment.get_best_models()
    except:
        'error in models retrieving'

    try:
        experiment.print_model_info()
    except:
        'error in printing models'
