import pickle
import json
import numpy as np

__data_columns = None
__locations = None
__model = None


def get_estimated_price(location, sqft, bath, bhk):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1
    X = np.zeros(len(__data_columns))
    X[0] = sqft
    X[1] = bath
    X[2] = bhk
    if loc_index >= 0:
        X[loc_index] = 1
    return round(float(__model.predict([X])[0]), 2)


def get_location_names():
    return __locations


def load_saved_artifacts():
    global __data_columns
    global __locations
    global __model
    print("Loading saved artifacts...Start")
    with open("artifacts/columns.json", "r") as file:
        __data_columns = json.load(file)["data_columns"]
        __locations = __data_columns[3:]

    with open("artifacts/model.pickle", "rb") as file:
        __model = pickle.load(file)
    print("Loading saved artifacts...Done")
