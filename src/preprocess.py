import pickle
import random
from kdtree import *
from read_pdb_file import read_pdb
from tqdm import tqdm

# try M proteins and N ligands
# Build M trees for protein, and try M*N queries for each pair
# note: n = (i-1)*N + j-1
M = 1
NUM_NEAR = 3


def split(ratio):
    return int(3000 * ratio)


def extract_data(i, type):
    x_list, y_list, z_list, atomtype_list = read_pdb('../data/training_data/%s_%s_cg.pdb' % (str(i).zfill(4), type))
    data = np.transpose(np.array([x_list, y_list, z_list, atomtype_list]))
    return np.array(data)


def store_tree():
    tree_list = []
    with open('../data/middle_data/tree_list.bin', 'wb') as f:
        for i in tqdm(range(1, 3000 + 1)):
            d_p = extract_data(i, 'pro')
            tree_list.append(build_KDTree(d_p))
        pickle.dump(tree_list, f)
        
    print('Info stored successfully!')


def sample_neg(num, N):
    # Sample n different ligs
    flag = True
    while flag:
        a = range(1, N+1)
        neg_sample = random.sample(a, num)
        if i not in neg_sample:
            flag = False
    return neg_sample



if __name__ == '__main__':
    N = split(0.9)

    # store_tree()
    f = open('../data/middle_data/tree_list.bin', 'rb')
    tree_list = pickle.load(f)
    print('Info loaded successfully!')

    # training data: 1 matched pair and 5 random unmatched pairs
    train_input = []
    train_output = []
    for i in tqdm(range(1, N+1)):
        pro = extract_data(i, 'pro')
        lig = extract_data(i, 'lig')

        train_input.append(find_nearest_atoms_KDTree(tree_list[i-1], lig, NUM_NEAR))
        train_output.append(1)

        neg_samples = sample_neg(5, N)
        for j in neg_samples:
            neg_lig = extract_data(j, 'lig')
            train_input.append(find_nearest_atoms_KDTree(tree_list[i-1], neg_lig, NUM_NEAR))
            train_output.append(-1)
    print('\ntraining data constructed successfully!\n')

    # validation data: store every pair
    valid_input = []
    valid_output = []
    for i in tqdm(range(N+1, 3001)):
        for j in range(N+1, 3001):
            lig = extract_data(j, 'lig')
            valid_input.append(find_nearest_atoms_KDTree(tree_list[i - 1], lig, NUM_NEAR))

            if i == j:
                valid_output.append(1)
            else:
                valid_output.append(-1)
    print('\nvalidation data constructed successfully!\n')



    #         print('\n\n*****************************************\nThe KDTree answer is:\n')
    #         output = find_nearest_atoms_KDTree(t, lig, 3)
    #         print(np.array(output))
    #
    #         print("\n\n*****************************************\nThe YMT's answer is:\n")
    #         print(find_nearest_atoms_YMT(pro, lig, 3))