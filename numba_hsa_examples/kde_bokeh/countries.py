import os
import pickle
from collections import defaultdict
from pprint import pprint

DATA_PATH = os.path.join(os.path.dirname(__file__), 'countries.pickle')


def load_data():
    with open(DATA_PATH, 'rb') as fin:
        data = pickle.load(fin)

    return data


def groupby_country(data):
    xlist = data['xlist']
    ylist = data['ylist']
    names = data['Ctry']

    out = defaultdict(list)
    for xs, ys, name in zip(xlist, ylist, names):
        assert type(name) is list
        assert len(name) == 1
        name = name[0]
        out[name].append({'xs': xs, 'ys': ys})

    return out


def main():
    dct = groupby_country(load_data())
    pprint(dct.keys())


if __name__ == '__main__':
    main()
