import keras
from keras.models import Sequential, Model
from keras import layers
from keras.optimizers import SGD


def build_source_cnn():
    length = 75

    input_layer = layers.Input(name='input', shape=(length ,1))

    x = layers.Conv1D(64, 3,padding='same', name='conv_1', kernel_regularizer = 'l2')(input_layer)
    x = layers.MaxPooling1D(2)(x,name='maxpooling_1')
   

    x = layers.Conv1D(128, 3,padding='same', name='conv_2', kernel_regularizer = 'l2')(x)
    feature = layers.MaxPooling1D(2)(x,name='maxpooling_2')


    y = layers.Flatten()(feature,name='flatten')
    y  = layers.Dense(128, activation="relu")(y,name='dense_1')
    bottle1 = layers.Dense(64, activation="relu")(y,name='dense_2')

    final_classifier  = layers.Dense(5, activation="sigmoid")(bottle1,name='final_classifier')
    source_cnn = Model(inputs=input_layer, outputs=final_classifier)
    
    source_feature=Model(inputs=input_layer,outputs=bottle1)
    
    return source_cnn

  
def build_source_cnn():
    length = 75

    input_layer = layers.Input(name='input', shape=(length ,1))

    x = layers.Conv1D(64, 3,padding='same', name='conv_1', kernel_regularizer = 'l2')(input_layer)
    x = layers.MaxPooling1D(2)(x,name='maxpooling_1')
   

    x = layers.Conv1D(128, 3,padding='same', name='conv_2', kernel_regularizer = 'l2')(x)
    feature = layers.MaxPooling1D(2)(x,name='maxpooling_2')


    y = layers.Flatten()(feature,name='flatten')
    y  = layers.Dense(128, activation="relu")(y,name='dense_1')
    bottle1 = layers.Dense(64, activation="relu")(y,name='dense_2')

    final_classifier  = layers.Dense(5, activation="sigmoid")(bottle1,name='final_classifier')
    target_cnn = Model(inputs=input_layer, outputs=final_classifier)
    
    target_feature=Model(inputs=input_layer,outputs=bottle1)
    
    return target_cnn
  
  
def build_domain_classifier:
    length=64
    input_layer = layers.Input(name='input', shape=(length, ))
    d = layers.Dense(1, activation='sigmoid')(input_layer,name='classifier')
    discriminator = Model(inputs=input_layer, outputs=d)
    return discriminator
  
  
def build_domain_classifier_0:
    length=64
    input_layer = layers.Input(name='input', shape=(length, ))
    x = layers.Dense(10, activation='relu')(x,name='dense')
    d = layers.Dense(1, activation='sigmoid')(x,name='classifier')
    discriminator = Model(inputs=input_layer, outputs=d)
    return discriminator




  
