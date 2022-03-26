import csv
import os
import pickle
import argparse

from util import AttributeDict

def data_loader(data_path):
    data_dict = {}
    with open(data_path, 'r') as f:
        csv_reader = csv.reader(f, delimiter='\t')
        for s,r,o in csv_reader:
            try:
                data_dict[s][r].append(o)
            except KeyError:
                if data_dict.get(s) is None:
                    data_dict[s] = dict()
                data_dict[s][r] = [o]
    return data_dict

def build_dataset(data_dict):
    x, y = list(), list()
    e2index, index2e, r2index, index2r = dict(), dict(), dict(), dict()
    for s, ro in data_dict.items():
        try:
            _ = e2index[s]
        except KeyError:
            index = len(e2index)
            e2index[s] = index
            index2e[index] = s
        
        for r, os in ro.items():
            try:
                _ = r2index[r]
            except KeyError:
                index = len(r2index)
                r2index[r] = index
                index2r[index] = r
            
            for o in os:
                try:
                    _ = e2index[o]
                except KeyError:
                    index = len(e2index)
                    e2index[o] = index
                    index2e[index] = o
            
            x.append((s,r))
            y.append(os)
                    
    return x, y, e2index, index2e, r2index, index2r


def preprocess_train(data_path):
    data_dict = data_loader(data_path)
    x, y, e2index, index2e, r2index, index2r = build_dataset(data_dict)
    
    data = {
        'x': x,
        'y': y,
        'e2index': e2index,
        'index2e': index2e,
        'r2index': r2index,
        'index2r': index2r
    }
    
    print("#entities:{0} ".format(len(e2index)))
    print("#relations:{0} ".format(len(r2index)))
    
    save_data_path = os.path.splitext(data_path)[0] + '.pkl'
    pickle.dump(data, open(save_data_path, 'wb'))



def preprocess_valid(train_path, valid_path):
    x, y = list(), list()
    with open(train_path, 'rb') as f:
        train_data = AttributeDict(pickle.load(f))
    
    data_dict = data_loader(valid_path)
    
    for s, ro in data_dict.items():
        try:
            _ = train_data.e2index[s]
        except KeyError:
            continue
        
        for r, objects in ro.items():
            try:
                _ = train_data.r2index[r]
            except KeyError:
                continue
            
            filtered_objects = list()
            
            for o in objects:
                try:
                    _ = train_data.e2index[o]
                    filtered_objects.append(o)
                except KeyError:
                    continue
            
            x.append((s,r))
            y.append(filtered_objects)
    
    data = {
        'x':x,
        'y':y,
    }
    
    save_file_path = os.path.splitext(valid_path)[0] + '.pkl'
    pickle.dump(data, open(save_file_path, 'wb'))
    
    
def parse_args():
    parser = argparse.ArgumentParser(description = 'Preprocess knowledge graph csv/txt train/valid data.')
    sub_parsers = parser.add_subparsers(help='mode', dest='mode')
    sub_parsers.required = True
    train_parser = sub_parsers.add_parser('train', help='Preprocess a training set')
    valid_parser = sub_parsers.add_parser('valid', help='Preprocess a valid or test set')
    
    train_parser.add_argument('train_path', type=str, help='Path to the raw train dataset (csv or txt file)')
    
    valid_parser.add_argument('train_path', type=str, help='Path to preprocessed train dataset (pkl file)')
    valid_parser.add_argument('valid_path', type=str, help='Path to raw valid dataset (csv or txt file)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == 'train':
        preprocess_train(args.train_path)
    else:
        preprocess_valid(args.train_path, args.valid_path)

if __name__ == '__main__':
    main()