from multiprocessing import cpu_count
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.callbacks import EarlyStopping


def define_models():
    models = {
        'RandomForest': [
            ['random_forest_1',
             dict(
                 n_estimators=3,
                 max_depth=2,
                 criterion='gini',
                 n_jobs=cpu_count()
             )],

        ],

        'NN':
             [
                 ['FFNN_2',
                  (
                      ([
                           Input(shape=(298,)),
                           Dense(1, activation='sigmoid')
                       ],),

                      dict(
                          optimizer='nadam',
                          loss='binary_crossentropy'
                      ),

                      dict(
                          epochs=1,
                          batch_size=1024,
                          validation_split=0.1,
                          shuffle=True,
                          verbose=True,
                          callbacks=[
                              EarlyStopping(monitor='val_loss', mode='min'),
                          ]
                      )

                  )]
             ]
    }
    defined_models = {}
    for algorithm in models:
        defined_models[algorithm] = []
        for model in models.get(algorithm):
            defined_models.get(algorithm).append(model)
    return defined_models
