from bioinformatica.source.preprocessing.decompositions import *
from bioinformatica.source.datasets.loader import get_data


def test_pca_function():
    parameters = ('K562', 200, 'enhancers'), 'epigenomic'
    dataset, labels = get_data(parameters)
    dataset = dataset.head(100)
    dataset = PCA_function(dataset)
    assert dataset is not None, 'an error occurred while decomposing the dataset with PCA'


def test_tsne_function():
    parameters = ('K562', 200, 'enhancers'), 'epigenomic'
    dataset, labels = get_data(parameters)
    dataset = dataset.head(100)
    dataset = TSNE_function(dataset)
    assert dataset is not None, 'an error occurred while decomposing the dataset with TSNE'



