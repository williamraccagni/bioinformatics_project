from bioinformatica.source.datasets.utils import holdouts, load_dataset
import numpy as np
from ucsc_genomes_downloader import Genome
from keras_bed_sequence import BedSequence
from keras_mixed_sequence import MixedSequence
from bioinformatica.source.type_hints import *
import pandas as pd


def get_data(parameters: Tuple[Tuple[str, int, str], str]) -> Tuple[pd.DataFrame, np.array] or List[np.array, np.array]:
    load_parameters, data_type = parameters
    if data_type == 'epigenomic':
        dataset, labels = load_dataset(load_parameters)
        dataset.reset_index(drop=True, inplace=True)
        return dataset, labels
    if data_type == 'sequences':
        epigenomes, labels = load_dataset(load_parameters)
        genome = Genome('hg19')
        bed = epigenomes.reset_index()[epigenomes.index.names]
        batch_size = len(labels)
        return [data for data in MixedSequence(
                x=BedSequence(genome, bed.iloc[np.arange(batch_size)], batch_size=batch_size),
                y=labels[np.arange(batch_size)],
                batch_size=batch_size)[0]]


def get_holdouts(dataset: pd.DataFrame or np.array, labels: np.array, holdout_parameters: Tuple[int, float, int], data_type: str) \
        -> Tuple[Tuple[pd.DataFrame, np.array], Tuple[pd.DataFrame, np.array]] or Tuple[Tuple[np.array, np.array], Tuple[np.array, np.array]]:
    if data_type == 'epigenomic':
        for training_indexes, test_indexes in holdouts(holdout_parameters).split(dataset, labels):
            yield ((dataset.iloc[training_indexes], labels[training_indexes]),
                   (dataset.iloc[test_indexes], labels[test_indexes]))
    else:
        for training_indexes, test_indexes in holdouts(holdout_parameters).split(dataset, labels):
            yield ((dataset.iloc[training_indexes], labels[training_indexes]),
                   (dataset.iloc[test_indexes], labels[test_indexes]))
