from bioinformatica.source.models.libraries import *
from bioinformatica.source.type_hints import *
import numpy as np


def build_DecisionTree(hyperparameters: Dict) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(**hyperparameters)


def build_SGD(hyperparameters: Dict) -> SGDClassifier:
    return SGDClassifier(**hyperparameters)


def build_RandomForest(hyperparameters: Dict) -> RandomForestClassifier:
    return RandomForestClassifier(**hyperparameters)


def build_NeuralNetwork(parameters: Dict) -> Sequential:
    network_parameters, compiling_parameters = parameters
    model = Sequential(*network_parameters)
    model.compile(**compiling_parameters)
    return model


def train_model(is_NN: bool, model, training_data: Tuple[np.array, np.array] or Tuple[np.array, np.array, Dict]) -> \
        None or ...:
    if is_NN:
        X_train, y_train, training_parameters = training_data
        model.fit(X_train, y_train, **training_parameters)
    else:
        X_train, y_train = training_data
        return model.fit(X_train, y_train)
