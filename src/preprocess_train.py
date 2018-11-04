import pickle
import random
from kdtree import *
from read_pdb_file import read_pdb
from tqdm import tqdm

NUM_NEG = 2
NUM_NEAR = 4


def extract_data(i, type):
    x_list, y_list, z_list, atomtype_list = read_pdb('../data/training_data/%s_%s_cg.pdb' % (str(i).zfill(4), type))
    data = np.transpose(np.array([x_list, y_list, z_list, atomtype_list]))
    return np.array(data)


def find_mean_point(data):
    '''

    :param data: [[],[],[]]
    :return: []
    '''
    mean_v = np.mean(data, axis=0)
    vec_o = np.delete(mean_v, 3)
    return vec_o


def sample_neg(num, i, N):
    # Sample n different ligs
    flag = True
    while flag:
        a = range(1, N + 1)
        neg_sample = random.sample(a, num)
        if i not in neg_sample:
            flag = False
    return neg_sample



############### methods used in cnn ###############
def normalize_centroid_throwtype(data, meanpoint):
    transform_data = data[:, :3] - meanpoint

    # scaling_data = np.trunc(transform_data / 3).astype(np.int)
    return transform_data


def prepare_CNN(index, type):
    if type is 'pro':
        pro = extract_data(index, 'pro')
        origin_point_pro = find_mean_point(pro)
        pro = normalize_centroid_throwtype(pro, origin_point_pro)
        # print(origin_point_pro)
        cnn_pro = np.zeros((27, 27, 27, 1))
        for atom in pro:
            location = list(map(lambda x: int(x / 5), atom))
            x = location[0]
            y = location[1]
            z = location[2]
            if x in range(-13, 14) and y in range(-13, 14) and z in range(-13, 14):
                cnn_pro[x][y][z][0] += 1
            else:
                pass
        return cnn_pro
    elif type is 'lig':
        lig = extract_data(index, 'lig')
        origin_point_lig = find_mean_point(lig)
        lig = normalize_centroid_throwtype(lig, origin_point_lig)
        # print(origin_point_lig)
        cnn_lig = np.zeros((6, 6, 6, 1))
        for atom in lig:
            location = list(map(lambda x: int(x / 5), atom))
            x = location[0]
            y = location[1]
            z = location[2]
            if x in range(-3, 3) and y in range(-3, 3) and z in range(-3, 3):
                cnn_lig[x][y][z][0] += 1
            else:
                pass
        return cnn_lig
    else:
        print('Wrong type!')


def create_CNN_train(num):
    cnn_pro_train = []
    cnn_lig_train = []

    print('Begin storing train dataset')
    for i in tqdm(range(1, num + 1)):
        cnn_pro_train.append(prepare_CNN(i, 'pro'))
        cnn_lig_train.append(prepare_CNN(i, 'lig'))



    with open('../data/cnn_data/cnn_pro_train.bin', 'wb') as f:
        pickle.dump(cnn_pro_train, f)
    with open('../data/cnn_data/cnn_lig_train.bin', 'wb') as f:
        pickle.dump(cnn_lig_train, f)
    print('\nCNN training data stored successfully!\n')




'''
def create_CNN_train(num):
    cnn_pro_train = []
    cnn_lig_train = []
    cnn_out_train = []

    for i in tqdm(range(1, num + 1)):
        cnn_pro_train.append(prepare_CNN(i, 'pro'))
        cnn_lig_train.append(prepare_CNN(i, 'lig'))
        cnn_out_train.append([1])

        indexes_neg = sample_neg(NUM_NEG, i, 2700)
        for sample in indexes_neg:
            cnn_pro_train.append(prepare_CNN(i, 'pro'))
            cnn_lig_train.append(prepare_CNN(sample, 'lig'))
            cnn_out_train.append([-1])

    with open('../data/cnn_data/cnn_pro_train.bin', 'wb') as f:
        pickle.dump(cnn_pro_train, f)
    with open('../data/cnn_data/cnn_lig_train.bin', 'wb') as f:
        pickle.dump(cnn_lig_train, f)
    with open('../data/cnn_data/cnn_out_train.bin', 'wb') as f:
        pickle.dump(cnn_out_train, f)
    print('\nCNN training data stored successfully!\n')


def create_CNN_valid(num1, num2):
    cnn_pro_valid = []
    cnn_lig_valid = []
    cnn_out_valid = []

    print('Begin storing valid dataset')
    for i in tqdm(range(num1 + 1, num2 + 1)):
        for j in range(num1 + 1, num2 + 1):
            cnn_pro_valid.append(prepare_CNN(i, 'pro'))
            cnn_lig_valid.append(prepare_CNN(j, 'lig'))
            if i == j:
                cnn_out_valid.append([1])
            else:
                cnn_out_valid.append([-1])

    with open('../data/cnn_data/cnn_pro_valid.bin', 'wb') as f:
        pickle.dump(cnn_pro_valid, f)
    with open('../data/cnn_data/cnn_lig_valid.bin', 'wb') as f:
        pickle.dump(cnn_lig_valid, f)
    with open('../data/cnn_data/cnn_out_valid.bin', 'wb') as f:
        pickle.dump(cnn_out_valid, f)
    print('\nCNN training data stored successfully!\n')
'''









############### methods used in lstm/mlp ###############
def normalize_centroid_keeptype(data, meanpoint):
    mean = np.append(meanpoint, 0)
    t = data - mean
    return t


def store_tree():
    tree_list = []
    with open('../data/middle_data/tree_list.bin', 'wb') as f:
        for i in tqdm(range(1, 3000 + 1)):
            pro = extract_data(i, 'pro')
            origin_point_pro = find_mean_point(pro)
            pro = normalize_centroid_keeptype(pro, origin_point_pro)

            tree_list.append(build_KDTree(pro))
        pickle.dump(tree_list, f)
    print('Info stored successfully!')
    return tree_list



def create_mlp_lstm_train(tree_list, N):
    # training data: 1 matched pair and 5 random unmatched pairs
    train_input = []
    train_output = []
    for i in tqdm(range(1, N + 1)):
        pro = extract_data(i, 'pro')
        # get pro centroid
        origin_point_pro = find_mean_point(pro)
        # recompute lig coordinates
        lig = extract_data(i, 'lig')
        lig = normalize_centroid_keeptype(lig, origin_point_pro)

        train_input.append(find_nearest_atoms_KDTree(tree_list[i - 1], lig, NUM_NEAR))
        train_output.append([1])

        neg_samples = sample_neg(NUM_NEG, i, N)
        for j in neg_samples:
            neg_lig = extract_data(j, 'lig')
            neg_lig = normalize_centroid_keeptype(neg_lig, origin_point_pro)
            train_input.append(find_nearest_atoms_KDTree(tree_list[i - 1], neg_lig, NUM_NEAR))
            train_output.append([-1])

    with open('../data/middle_data/train_input.bin', 'wb') as f:
        pickle.dump(train_input, f)
    with open('../data/middle_data/train_output.bin', 'wb') as f:
        pickle.dump(train_output, f)
    print('\ntraining data stored successfully!\n')


def create_mlp_lstm_valid(tree_list, begin=2700, end=3000):
    # validation data: store every pair
    valid_input = []
    valid_output = []
    for i in tqdm(range(begin+1, end+1)):
        pro = extract_data(i, 'pro')
        # get pro centroid
        origin_point_pro = find_mean_point(pro)

        for j in range(begin+1, end+1):
            # recompute lig coordinates
            lig = extract_data(j, 'lig')
            lig = normalize_centroid_keeptype(lig, origin_point_pro)
            valid_input.append(find_nearest_atoms_KDTree(tree_list[i - 1], lig, NUM_NEAR))

            if i == j:
                valid_output.append([1])
            else:
                valid_output.append([-1])

    print('\nvalidation data constructed successfully!\n')
    return valid_input, valid_output


if __name__ == '__main__':

    treelist = store_tree()
    create_mlp_lstm_train(tree_list, 2700)
    create_CNN_train(3000)




    








