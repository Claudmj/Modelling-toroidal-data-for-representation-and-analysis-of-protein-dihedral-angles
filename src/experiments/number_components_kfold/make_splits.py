import os
import json
import numpy as np
from sklearn.model_selection import KFold
from src.paths import *

NUM_SPLITS = 5
JSON_FILE_PATH = os.path.join(DATA_DIRECTORY, '9mers.json')

with open(JSON_FILE_PATH, "rt") as file:
    amino_acids = json.load(file)

for amino_acid in amino_acids:
    angles = amino_acids[amino_acid]
    train_test_dict = {}
    kfold = KFold(n_splits=NUM_SPLITS, random_state=123, shuffle=True)
    shuffle = [(np.random.choice(a=train, size=1000, replace=False).tolist(), np.random.choice(a=test, size=1000, replace=False).tolist()) for train, test in kfold.split(angles)]

    file_path = os.path.join(DATA_DIRECTORY, "splits", f"{amino_acid}_{NUM_SPLITS}_split.json")
    with open(file_path, "wt") as file:
        file.write(json.dumps(shuffle))

