from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score
from pprint import pprint


statistical_tests = [
    wilcoxon
]

metrics = [
    (accuracy_score, 'labels'),
    (average_precision_score, 'probabilistic'),
    (balanced_accuracy_score, 'labels'),
    (roc_auc_score, 'probabilistic')
]


def print_model(model):
    print(model.get_name())
    pprint(model.get_scores())
    print('\n')
