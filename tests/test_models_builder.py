from bioinformatica.source.models.builder import *
from multiprocessing import cpu_count
from bioinformatica.source.datasets.utils import load_dataset
from bioinformatica.source.experiments.utils import metrics


def test_models():
    dataset, labels = load_dataset(('K562', 200, 'enhancers'))
    dataset = dataset.head(100)
    labels = labels[:100]
    n_sample = len(dataset)
    train_split = int(n_sample * 0.8)
    test_split = n_sample - train_split
    X_train, y_train = dataset.head(train_split), labels[:train_split]
    X_test, y_test = dataset.tail(test_split), labels[train_split:]
    training_data = (X_train, y_train)
    non_nn_parameters = dict(
        n_estimators=20,
        max_depth=5,
        criterion='gini',
        n_jobs=cpu_count()
    )
    nn_parameters = (
        ([
            Input(shape=(298, )),
            Dense(1, activation="sigmoid")
        ], 'FFNN'),

        dict(optimizer="nadam",
             loss="binary_crossentropy"),

        dict(
            epochs=1,
            batch_size=1024,
            validation_split=0.1,
            shuffle=True,
            verbose=False,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min"),
            ]
        )
    )
    non_nn_model = Model('RandomForest', 'RF1', False)
    nn_model = Model('NN', 'FFNN1', True)
    non_nn_model.build(non_nn_parameters), 'an error occurred while building non nn model'
    nn_model.build(nn_parameters), 'an error occurred while building nn model'
    assert non_nn_model.get_model() is not None, 'no instance model is defined for this non nn model'
    assert nn_model.get_model() is not None, 'no instance model is defined for this nn model'

    assert len(vars(non_nn_model)) == 7, 'non NN model was incorrectly initialized'
    assert len(vars(nn_model)) == 8, 'NN model was incorrectly initialized'
    non_nn_model.train(training_data)
    nn_model.train(training_data)
    assert non_nn_model.get_trained_model() is not None, 'no training instance model is defined for this non nn model'
    assert nn_model.get_trained_model() is not None, 'no training instance model is defined for this nn model'

    non_nn_prediction = non_nn_model.predict(X_test)
    nn_prediction = nn_model.predict(X_test)

    assert non_nn_prediction is not None, 'an error occurred while predicting new values for non nn model'
    assert nn_prediction is not None, 'an error occurred while predicting new values for nn model'

    for metric in metrics:
        non_nn_model.test_metrics(metric, (y_test, non_nn_prediction))
        nn_model.test_metrics(metric, (y_test, nn_prediction))

    assert len(non_nn_model.get_scores()) == len(metrics), 'some metrics could not be executed for this non nn model'
    assert len(non_nn_model.get_scores()) == len(metrics), 'some metrics could not be executed for this nn model'

    try:
        non_nn_model.get_name()
    except:
        'no name for non nn model'
    try:
        nn_model.get_name()
    except:
        'no name for nn model'
