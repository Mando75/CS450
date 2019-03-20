import pandas as pd


def load_flare():
    flare_data = pd.read_csv(
        "../datasets/flare.data2.csv", header=None, delimiter=" ")
    flare_data[1] = [ord(x) - 64 for x in flare_data[1]]
    flare_data[2] = [ord(x) - 64 for x in flare_data[2]]
    struct = {}
    struct["targets"] = [ord(x) - 64 for x in flare_data[0]]
    struct["data"] = flare_data.drop(0, axis=1)
    return struct
