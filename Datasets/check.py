import pickle

data_path = '/home/server12gb/Desktop/Bach/UparChallenge_baseline/pipeline/Conformer/Datasets/val_all.pkl'
dataset_info = pickle.load(open(data_path, 'rb+'))
print(dataset_info)