""" Covert the digitStruct.mat to json.
Source: https://stackoverflow.com/questions/41176258/h5py-access-data-in-datasets-in-svhn
"""
import os
import h5py
import numpy as np
import tqdm
import json


ROOT = "/data/Data2/Public-Folders/SVHN"


def get_box_data(index, hdf5_data):
    meta_data = dict()
    meta_data['height'] = []
    meta_data['label'] = []
    meta_data['left'] = []
    meta_data['top'] = []
    meta_data['width'] = []

    def print_attrs(name, obj):
        vals = []
        if obj.shape[0] == 1:
            vals.append(obj[0][0])
        else:
            for k in range(obj.shape[0]):
                vals.append(int(hdf5_data[obj[k][0]][0][0]))
        meta_data[name] = vals

    box = hdf5_data['/digitStruct/bbox'][index]
    hdf5_data[box[0]].visititems(print_attrs)
    return meta_data


def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])


def load(split):
    mat_data = h5py.File(os.path.join(ROOT, split, 'digitStruct.mat'), 'r')
    size = mat_data['/digitStruct/name'].size

    data = []
    for _i in tqdm.tqdm(range(size)):
        pic = os.path.join(ROOT, split, get_name(_i, mat_data))
        box = get_box_data(_i, mat_data)
        data.append({'path':pic, 'anno':box})

    with open('%s.json'%split, 'w') as f:
        json.dump(data, f)


def main():
    load('train')
    load('extra')
    load('test')


if __name__ == '__main__':
    main()

