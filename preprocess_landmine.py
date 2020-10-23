import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

landmine_data = sio.loadmat("LandmineData.mat")

features = landmine_data["feature"][0]
label = landmine_data["label"][0]

all_X_train = []
all_Y_train = []
all_X_test = []
all_Y_test = []

for i in range(len(features)):
    X_train, X_test, Y_train, Y_test = train_test_split(features[i], label[i], test_size=0.5, stratify=label[i], random_state=0)

    all_X_train.append(X_train)
    all_Y_train.append(Y_train)
    all_X_test.append(X_test)
    all_Y_test.append(Y_test)
all_data = {"all_X_train":all_X_train, "all_Y_train":all_Y_train, "all_X_test":all_X_test, "all_Y_test":all_Y_test}

pickle.dump(all_data, open("landmine_formated_data.pkl", "wb"))

