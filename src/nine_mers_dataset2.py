import os
import json

from src.paths import DATA_DIRECTORY

class NineMersDataset2:
    def __init__(self):
        self.AMINO_ACIDS = [
            "M",
            "N",
            "I",
            "F",
            "E",
            "L",
            "R",
            "D",
            "G",
            "K",
            "Y",
            "T",
            "H",
            "S",
            "P",
            "A",
            "V",
            "Q",
            "W",
            "C",
        ]

        with open(os.path.join(DATA_DIRECTORY, '9mers.json'), "rt") as file:
            self.dataset = json.load(file)
