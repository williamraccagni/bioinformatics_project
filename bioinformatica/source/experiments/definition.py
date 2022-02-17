from bioinformatica.source.models.libraries import *
from multiprocessing import cpu_count
from bioinformatica.source.type_hints import *

import tensorflow as tf

'''
This functions allows to return a dictionary containing all the models to be trained. It returns a dictionary: every key
is a string that represents a machine learning algorithm, the value for each key is a list that contains one or more models
for a particular algorithm. Every model is represented as a list of the values: a string for the name of the model and a 
tuple containing the parameters to build and train a model.
Neural networks use a tuple of three elements: the parameters to construct the network, compiling parameters and training parameters:

Example of use for K562 enhancers epigenomic dataset:

models = {
        'RandomForest': [
            ['random_forest_1',
             dict(
                 n_estimators=20,
                 max_depth=5,
                 criterion='gini',
                 n_jobs=cpu_count()
             )],
             
             ['random_forest_2',
             dict(
                 n_estimators=20,
                 max_depth=5,
                 criterion='gini',
                 n_jobs=cpu_count()
             )],
             
             ['random_forest_3',
             dict(
                 n_estimators=20,
                 max_depth=5,
                 criterion='gini',
                 n_jobs=cpu_count()
             )]

        ],

        'NN':

             [
             ['FFNN_1',
              (
                  ([
                       Input(shape=(298,)),
                       Dense(32, activation='relu'),
                       Dense(16, activation='relu'),
                       Dense(1, activation='sigmoid')
                   ],),

                  dict(
                      optimizer='nadam',
                      loss='binary_crossentropy'
                  ),

                  dict(
                      epochs=10,
                      batch_size=1024,
                      validation_split=0.1,
                      shuffle=True,
                      verbose=True,
                      callbacks=[
                          EarlyStopping(monitor='val_loss', mode='min'),
                      ]
                  )

              )],

             ['FFNN_2',
              (
                  ([
                       Input(shape=(298,)),
                       Dense(32, activation='relu'),
                       Dense(16, activation='relu'),
                       Dense(1, activation='sigmoid')
                   ],),

                  dict(
                      optimizer='nadam',
                      loss='binary_crossentropy'
                  ),

                  dict(
                      epochs=10,
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
'''



def define_models() -> Dict[str, List]:

    # #SEQUENCES
    #
    # Input_layer = Input(shape=(800,))
    #
    # activation_function = 'relu'
    #
    # models = {
    #
    #     'NN' :
    #     [
    #
    #         ["Perceptron",
    #          (
    #                  ([Input_layer,
    #                      Reshape((800, 1)),
    #                      Flatten(),
    #                      Dense(1, activation="sigmoid")
    #                  ],),
    #
    #
    #                  dict(optimizer="nadam",
    #                      loss="binary_crossentropy"),
    #
    #                  dict(
    #                      epochs=1000,
    #                      batch_size=1024,
    #                      validation_split=0.1,
    #                      shuffle=True,
    #                      verbose=False,
    #                      callbacks=[
    #                          EarlyStopping(monitor="val_loss", mode="min", patience=10),
    #                      ]
    #                  )
    #              )],
    #
    #             ["FFNN",
    #              (
    #                          ([
    #                               Input_layer,
    #                               Reshape((800, 1)),
    #                               Flatten(),
    #                               Dense(256, activation=activation_function),
    #                               Dense(128, activation=activation_function),
    #                               Dropout(0.3),
    #                               Dense(64, activation=activation_function),
    #                               Dropout(0.3),
    #
    #                               Dense(16, activation=activation_function),
    #                               Dense(1, activation="sigmoid")
    #                           ],),
    #
    #                          dict(
    #                              optimizer="nadam",
    #                              loss='binary_crossentropy'
    #                          ),
    #
    #                          dict(
    #                              epochs=1000,
    #                              batch_size=1024,
    #                              validation_split=0.1,
    #                              shuffle=True,
    #                              verbose=False,
    #                              # class_weight=class_weightz,
    #                              callbacks=[
    #                                  EarlyStopping(monitor="val_loss", mode="min", patience=10),
    #                              ]
    #                          )
    #
    #                      )],
    #             ["CNN_1",
    #              (
    #                  ([
    #                       Input_layer,
    #                       Reshape((200, 4, 1)),
    #                       Conv2D(32, kernel_size=(7, 1), activation="relu"),
    #                       Conv2D(32, kernel_size=(7, 1), activation="relu"),
    #                       MaxPooling2D((2, 2)),
    #                       Conv2D(16, kernel_size=(7, 1), activation="relu"),
    #                       Flatten(),
    #                       Dense(128, activation="relu"),
    #                       Dropout(0.2),
    #                       Dense(32, activation="relu"),
    #                       Dense(16, activation="relu"),
    #                       Dense(1, activation="sigmoid")
    #                   ],),
    #
    #                  dict(optimizer="nadam",
    #                       loss="binary_crossentropy"),
    #
    #                  dict(
    #                      epochs=1000,
    #                      batch_size=1024,
    #                      validation_split=0.1,
    #                      shuffle=True,
    #                      verbose=False,
    #                      callbacks=[
    #                          EarlyStopping(monitor="val_loss", mode="min", patience=5),
    #                      ]
    #                  )
    #
    #              )],
    #         ["CNN_2",
    #          (
    #              ([
    #                      Input_layer,
    #                      Reshape((200, 4, 1)),
    #                      Conv2D(64, kernel_size=(7, 2), activation="relu"),
    #                      Conv2D(64, kernel_size=(7, 2), activation="relu"),
    #                      MaxPooling2D((2, 2)),
    #                      Conv2D(32, kernel_size=(5, 1), activation="relu"),
    #                      UpSampling2D(),
    #                      Conv2D(16, kernel_size=(7, 1), activation="relu"),
    #                      Flatten(),
    #                      Dense(128, activation="relu"),
    #                      Dropout(0.2),
    #                      Dense(32, activation="relu"),
    #                      Dropout(0.2),
    #                      Dense(16, activation="relu"),
    #                      Dropout(0.2),
    #                      Dense(1, activation="sigmoid")
    #                  ],),
    #
    #              dict(optimizer="nadam",
    #                  loss="binary_crossentropy"),
    #
    #              dict(
    #                  epochs=1000,
    #                  batch_size=1024,
    #                  validation_split=0.1,
    #                  shuffle=True,
    #                  verbose=False,
    #                  callbacks=[
    #                      EarlyStopping(monitor="val_loss", mode="min", patience=5),
    #                  ]
    #              )
    #
    #          )]
    #
    #     ]
    #
    # }











    #EPIGENOMIC

    Input_layer = Input(shape=(104,)) #Input(shape=(298,))
    activation_function = "relu"

    initializer = 'random_uniform'
    regularizer = tf.keras.regularizers.l2(0.01) #tf.keras.regularizers.l1(0.01)

    init_bias = tf.keras.initializers.Zeros()
    # init_bias = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
    # learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)



    nn_input_dimension = 104
    nf = 50 #100
    x = Dense(units=nf, activation='relu', kernel_initializer=initializer)(Input_layer)
    x = Dense(units=nn_input_dimension, activation='relu', kernel_initializer=initializer)(x)
    encoded = Dense(units=nf, activation='relu', kernel_initializer=initializer)(x)
    encoder = Model(Input_layer, encoded)


    models = {

        'NN':
            [
                ["Perceptron",
                 (
                     ([
                        Input_layer,
                        Dense(1, activation="sigmoid")
                    ],),

                     dict(
                         optimizer='nadam',
                         loss='binary_crossentropy'
                     ),

                     dict(
                             epochs=1000,
                             batch_size=1024,
                             validation_split=0.1,
                             shuffle=True,
                             verbose=False,
                             callbacks=[
                                 EarlyStopping(monitor="val_loss", mode="min", patience=50),
                             ]
                         )

                 )],

                ['FFNN_1',
                 (
                     ([
                         Input_layer,
                         Dense(256, activation=activation_function),
                         Dense(128),
                         BatchNormalization(),
                         Activation('relu'),
                         Dense(64, activation=activation_function),
                         Dropout(0.3),
                         Dense(32, activation=activation_function),
                         Dense(16, activation=activation_function),
                         Dense(1, activation="sigmoid")
                     ],),

                     dict(
                         optimizer='nadam',
                         loss='binary_crossentropy'
                     ),

                     dict(
                        epochs=1000,
                        batch_size=1024,
                        validation_split=0.1,
                        shuffle=True,
                        verbose=True,
                        # class_weight=class_weightz,
                        callbacks=[
                            EarlyStopping(monitor="val_loss", mode="min", patience=10),
                        ]
                    )

                 )],

                ['FFNN_2',
                 (
                     ([
                         Input_layer,
                         Dense(512, activation=activation_function, kernel_initializer=initializer),
                         Dense(512, activation=activation_function, kernel_initializer=initializer),
                         Dropout(0.2),
                         Dense(256, activation=activation_function, kernel_initializer=initializer),
                         Dense(256, activation=activation_function, kernel_initializer=initializer),
                         Dropout(0.3),
                         Dense(128),
                         BatchNormalization(),
                         Activation(activation_function),
                         Dense(128, activation=activation_function, kernel_initializer=initializer),
                         Dense(64, activation=activation_function, kernel_initializer=initializer),
                         Dropout(0.3),
                         Dense(64, activation=activation_function, kernel_initializer=initializer),
                         Dense(16, activation=activation_function, kernel_initializer=initializer),
                         Dense(8, activation=activation_function, kernel_initializer=initializer),
                         Dense(1, activation="sigmoid")
                     ],),

                     dict(
                         optimizer='nadam',
                         loss='binary_crossentropy'
                     ),

                     dict(
                        epochs=1000,
                        batch_size=1024,
                        validation_split=0.1,
                        shuffle=True,
                        verbose=True,
                        callbacks=[
                            EarlyStopping(monitor="val_loss", mode="min", patience=50),
                        ]
                    )

                 )],

                ['encoder_FFNN',
                 (
                     ([
                         encoder,




                         Dense(256),
                         BatchNormalization(),
                         Activation('relu'),
                         Dense(128, activation=activation_function, kernel_initializer=initializer,
                               activity_regularizer=regularizer),




                         Dropout(0.2),
                         Dense(32, activation=activation_function, kernel_initializer=initializer,
                               activity_regularizer=regularizer),
                         Dropout(0.2),
                         Dense(16, activation=activation_function, kernel_initializer=initializer,
                               activity_regularizer=regularizer),
                         Dropout(0.2),
                         Dense(1, activation="sigmoid", bias_initializer=init_bias)
                     ],),

                     dict(
                         optimizer='nadam',
                         loss='binary_crossentropy'
                     ),

                     dict(
                        epochs=1000,
                        batch_size=512,
                        validation_split=0.1,
                        shuffle=True,
                        verbose=False,
                        # class_weight=class_weightz,
                        callbacks=[
                            EarlyStopping(monitor="val_loss", mode="min", patience=10, restore_best_weights=True),
                            # learning_rate_scheduler
                        ]
                    )

                 )]

            ],

        'SGD': [

            ['SGD',#'SGD_1',
             dict(
               loss='hinge',
               penalty='l2',
               n_jobs=cpu_count(),
               random_state=1,
               class_weight=None
             )
            ]#,
            # ['SGD_2',
            #  dict(
            #      loss='hinge',
            #      penalty='l1',
            #      n_jobs=cpu_count(),
            #      random_state=1,
            #      class_weight=None
            #  )
            #  ],
            # ['SGD_3',
            #  dict(
            #      loss='hinge',
            #      penalty='l2',
            #      n_jobs=cpu_count(),
            #      random_state=1,
            #      class_weight='balanced'
            #  )
            #  ],
            # ['SGD_4',
            #  dict(
            #      loss='modified_huber',
            #      penalty='l2',
            #      n_jobs=cpu_count(),
            #      random_state=1,
            #      class_weight=None
            #  )
            #  ],
            # ['SGD_5',
            #  dict(
            #      loss='perceptron',
            #      penalty='l2',
            #      n_jobs=cpu_count(),
            #      random_state=1,
            #      class_weight=None
            #  )
            #  ],
            # ['SGD_6',
            #  dict(
            #      loss='perceptron',
            #      penalty='elasticnet',
            #      n_jobs=cpu_count(),
            #      random_state=1,
            #      class_weight='balanced'
            #  )
            #  ]

        ],

        'RandomForest': [
            ['Random Forest',#'random_forest_1',
             dict(
                 random_state=1,
                 n_jobs=cpu_count(),

             )]#,
            # ['random_forest_1_e_mezzo',
            #  dict(
            #      random_state=1,
            #
            #      class_weight='balanced',
            #      n_jobs=cpu_count()
            #  )],
            #
            # ['random_forest_2',
            #  dict(
            #      random_state=1,
            #      n_estimators=20,
            #      max_depth=5,
            #      criterion='gini',
            #      n_jobs=cpu_count()
            #  )],
            #
            # ['random_forest_3',
            #  dict(
            #      random_state=1,
            #      n_estimators=40,
            #      max_depth=5,
            #      criterion='gini',
            #      n_jobs=cpu_count(),
            #      min_samples_leaf=20,
            #      min_samples_split=20
            #  )],
            #
            # ['random_forest_4',
            #  dict(
            #      random_state=1,
            #      n_estimators=100,
            #      max_depth=5,
            #      criterion='gini',
            #      n_jobs=cpu_count()
            #  )],
            #
            # ['random_forest_5',
            #  dict(
            #      random_state=1,
            #      n_estimators=20,
            #      max_depth=10,
            #      criterion='gini',
            #      n_jobs=cpu_count()
            #  )],
            #
            # ['random_forest_6',
            #  dict(
            #      random_state=1,
            #      n_estimators=40,
            #      max_depth=10,
            #      criterion='gini',
            #      n_jobs=cpu_count()
            #  )],
            #
            # ['random_forest_7',
            #  dict(
            #      random_state=1,
            #      n_estimators=100,
            #      max_depth=10,
            #      criterion='gini',
            #      n_jobs=cpu_count()
            #  )],
            #
            # ['random_forest_8',
            #  dict(
            #      random_state=1,
            #      n_estimators=20,
            #      max_depth=15,
            #      criterion='gini',
            #      n_jobs=cpu_count()
            #  )],
            #
            # ['random_forest_9',
            #  dict(
            #      random_state=1,
            #      n_estimators=40,
            #      max_depth=15,
            #      criterion='gini',
            #      n_jobs=cpu_count()
            #  )],
            #
            # ['random_forest_10',
            #  dict(
            #      random_state=1,
            #      n_estimators=100,
            #      max_depth=15,
            #      criterion='gini',
            #      n_jobs=cpu_count()
            #  )]

        ],

        'DecisionTree': [
            # ['decision_tree_1',
            #  dict(
            #
            #  )],
            # ['decision_tree_1 e mezzo',
            #  dict(
            #      class_weight = 'balanced'
            #  )],
            ['Decision Tree',#'decision_tree_1 e settantacinque',
             dict(
                 min_samples_leaf=20,
                 min_samples_split=20
             )]#,
            # ['decision_tree_2',
            #  dict(
            #      max_depth=5
            #  )],
            # ['decision_tree_3',
            #  dict(
            #      max_depth=10
            #  )],
            # ['decision_tree_4',
            #  dict(
            #      max_depth=15
            #  )],
            # ['decision_tree_5',
            #  dict(
            #      max_depth=50
            #  )]
        ]



    }

    defined_models = { }
    for algorithm in models:
        defined_models[algorithm] = []
        for model in models.get(algorithm):
            defined_models.get(algorithm).append(model)
    return defined_models
