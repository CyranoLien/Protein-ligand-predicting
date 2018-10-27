import numpy as np
import heapq
import random
from read_pdb_file import read_pdb

x_train = []
y_train = []
x_validation = []
y_validation = []

'''
    input: pro:[[]] lig:[[]] num: number of output atoms
    output: [[atom1, atom2, ... atomN, lig1], [atom1, atom2, ... atomN, lig2], ...]
'''
def find_nearest_atoms(pro, lig, num=2):
    pro_lig= []
    for i in range(len(lig)):
        data = []
        distance = np.sqrt((pro[:,0]-lig[i,0])**2+(pro[:,1]-lig[i,1])**2+(pro[:,2]-lig[i,2])**2)
        # here get the locations of the nearest atoms
        index_nearest_atoms = heapq.nsmallest(num, range(len(distance)), distance.take)
        for j in range(num):
            data.append(pro[index_nearest_atoms[j]])
        data.append(lig[i])
        pro_lig.append(data)
    pro_lig = np.array(pro_lig)
    return pro_lig

# split training data
for i in range(1, 2500):
        num1 = i
        num1 = "%04d" % num1
        X_list, Y_list, Z_list, atomtype_list = read_pdb('../data/training_data/%s_pro_cg.pdb' % num1)
        X_list2, Y_list2, Z_list2, atomtype_list2 = read_pdb('../data/training_data/%s_lig_cg.pdb' % num1)
        pro = np.array([X_list, Y_list, Z_list, atomtype_list])
        lig = np.array([X_list2, Y_list2, Z_list2, atomtype_list2])
        pro = np.transpose(pro)
        lig = np.transpose(lig)
        data = find_nearest_atoms(pro, lig)
        x_train.append(data)
        y_train.append(np.array([1]))

        # Sample 5 different ligs
        flag = True
        while flag:
            a = range(1, 2500)
            neg_sample = random.sample(a, 5)
            if i not in neg_sample:
                flag = False

        print(neg_sample)
        for j in range(len(neg_sample)):
            num2 = neg_sample[j]
            num2 = "%04d" % num2
            X_list2, Y_list2, Z_list2, atomtype_list2 = read_pdb('../data/training_data/%s_lig_cg.pdb' % num2)
            lig = np.array([X_list2, Y_list2, Z_list2, atomtype_list2])
            lig = np.transpose(lig)
            data = find_nearest_atoms(pro, lig)
            x_validation.append(data)
            y_validation.append(np.array([-1]))

# split validation data (501*501 samples)
for i in range(2500, 3001):
    for j in range(2500, 3001):
        num1 = i
        num1 = "%04d" % num1
        num2 = j
        num2 = "%04d" % num2
        X_list, Y_list, Z_list, atomtype_list = read_pdb('../data/training_data/%s_pro_cg.pdb' % num1)
        X_list2, Y_list2, Z_list2, atomtype_list2 = read_pdb('../data/training_data/%s_lig_cg.pdb' % num2)
        pro = np.array([X_list, Y_list, Z_list, atomtype_list])
        lig = np.array([X_list2, Y_list2, Z_list2, atomtype_list2])
        pro = np.transpose(pro)
        lig = np.transpose(lig)
        data = find_nearest_atoms(pro, lig)
        if i==j:
            x_validation.append(data)
            y_validation.append(np.array([1]))
        else:
            x_validation.append(data)
            y_validation.append(np.array([-1]))


