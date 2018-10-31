import pickle
import numpy as np
from kdtree import *
from tqdm import tqdm

NUM_NEAR = 4


def read_pdb2(filename):
    with open(filename, 'r') as file:
        strline_L = file.readlines()
    # print(strline_L)
    X_list = list()
    Y_list = list()
    Z_list = list()
    atomtype_list = list()
    for strline in strline_L:
        # removes all whitespace at the start and end, including spaces, tabs, newlines and carriage returns
        stripped_line = strline.strip()
        # print(stripped_line)

        splitted_line = stripped_line.split('\t')

        X_list.append(float(splitted_line[0]))
        Y_list.append(float(splitted_line[1]))
        Z_list.append(float(splitted_line[2]))

        atomtype = str(splitted_line[3])
        if atomtype == 'C':
            atomtype_list.append(0.001)  # 'h' means hydrophobic
        else:
            atomtype_list.append(-0.001)  # 'p' means polar


    return X_list, Y_list, Z_list, atomtype_list


def extract_data2(i, type):
    x_list, y_list, z_list, atomtype_list = read_pdb2('../data/testing_data/%s_%s_cg.pdb' % (str(i).zfill(4), type))
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


def transform_data(data, meanpoint):
    transform_data = data[:, :3] - meanpoint

    # scaling_data = np.trunc(transform_data / 3).astype(np.int)
    return transform_data


def transform_data_tree(data, meanpoint):
    mean = np.append(meanpoint, 0)
    t = data - mean
    return t


def prepare_CNN(index, type):
    if type is 'pro':
        pro = extract_data2(index, 'pro')
        origin_point_pro = find_mean_point(pro)
        pro = transform_data(pro, origin_point_pro)
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
        lig = extract_data2(index, 'lig')
        origin_point_lig = find_mean_point(lig)
        lig = transform_data(lig, origin_point_lig)
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


def create_CNN_test(num1, num2):
    cnn_pro_test = []
    cnn_lig_test= []


    print('Begin storing test dataset')
    for i in tqdm(range(num1+1, num2+1)):
        for j in range(num1+1, num2+1):
            cnn_pro_test.append(prepare_CNN(i, 'pro'))
            cnn_lig_test.append(prepare_CNN(j, 'lig'))


    with open ('../data/cnn_data/cnn_pro_test.bin', 'wb') as f:
        pickle.dump(cnn_pro_test, f)
    with open ('../data/cnn_data/cnn_lig_test.bin', 'wb') as f:
        pickle.dump(cnn_lig_test, f)

    print('\nCNN testing data stored successfully!\n')


def store_tree():
    tree_list = []
    with open('../data/middle_data/tree_list_test.bin', 'wb') as f:
        for i in tqdm(range(1, 824 + 1)):
            pro = extract_data2(i, 'pro')
            origin_point_pro = find_mean_point(pro)
            pro = transform_data_tree(pro, origin_point_pro)

            tree_list.append(build_KDTree(pro))
        pickle.dump(tree_list, f)

    print('Info stored successfully!')


def create_mlp_test(tree_list):
    # validation data: store every pair
    test_input = []

    for i in tqdm(range(1, 825)):
        pro = extract_data2(i, 'pro')
        # get pro centroid
        origin_point_pro = find_mean_point(pro)

        for j in range(1, 825):
            # recompute lig coordinates
            lig = extract_data2(j, 'lig')
            lig = transform_data_tree(lig, origin_point_pro)
            test_input.append(find_nearest_atoms_KDTree(tree_list[i - 1], lig, NUM_NEAR))

    print('\ntest data constructed successfully!\n')

    return test_input


if __name__ == '__main__':

    #create_CNN_test(0, 24)
    store_tree()
    with open('../data/middle_data/tree_list_test.bin', 'rb') as f:
        tree_list = pickle.load(f)
    print('Tree info loaded successfully!')
    #reate_mlp_test(tree_list)
