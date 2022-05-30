import os
import pickle

from math import pi, sin, cos, atan2, sqrt

import tensorflow as tf

deg2rad = pi / 180
rad2deg = 180 / pi

def HaversineDistace(lat1, lat2, lon1, lon2):
    """Calculate Distance between two point on earth"""
    r = 6371 # km
    
    lat1 = lat1 * deg2rad
    lat2 = lat2 * deg2rad
    lon1 = lon1 * deg2rad
    lon2 = lon2 * deg2rad

    h = sin( (lat2 - lat1)/2 )**2 + cos(lat1)*cos(lat2)*( sin( (lon2 - lon1)/2 )**2 )
    return 2*r*atan2( sqrt( h ), sqrt( 1-h ) )

def ListModels(models_directory: str):
    models_list = []
    dirs = os.listdir(models_directory)

    for item in dirs:
        item_path = os.path.join(models_directory, item)
        if os.path.isdir(item_path):
            models_list.append(item_path)

    return models_list

def SelectModelPrompt(models_directory):

    models_list = ListModels(models_directory)
    models_list.sort()

    print ("Found {} models inside {}:".format(
                                        len(models_list), 
                                        models_directory
                                    ))
    
    print ("index    Model-name")
    for i, models_path in enumerate(models_list):
        number = "[{}]. ".format(i).ljust(7, " ")
        print ("  {}{}".format(
                            number, 
                            os.path.basename (models_path)
                        ))
    index = input("Please input your model's index (e.g 0): ")
    index = int(index)

    print ("You selected model {}".format(os.path.basename (models_list[index])))

    return models_list[index]

def LoadModel(path_to_model):
    """Load Model and optionally it's history as well"""
    history_file = os.path.join(path_to_model, 'history.pkl')
    model = tf.keras.models.load_model(path_to_model)
    # model = tf.saved_model.load(path_to_model)
    print ("model loaded")

    with open(history_file, 'rb') as f:
        history = pickle.load(f)
    print ("model history loaded")

    return model, history