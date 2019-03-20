import pandas as pd


def load_letter():
    letter_data = pd.read_csv(
        "../datasets/letter-recognition.data.csv", header=None, delimiter=',')
    struct = {}
    struct["targets"] = [ord(x) - 64 for x in letter_data[0]]
    struct["data"] = letter_data.drop(0, axis=1)
    return struct
