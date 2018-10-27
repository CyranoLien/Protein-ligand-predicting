import numpy as np
import heapq
from read_pdb_file import read_pdb
from scipy.spatial import KDTree

# try M proteins and N ligands
# Build M trees for protein, and try M*N queries for each pair
M = N = 1

def extract_data():
    pro_list = []
    lig_list = []
    for i in range(1, M+1):
        for j in range(1, N+1):
            X_list, Y_list, Z_list, atomtype_list = read_pdb('../data/training_data/%s_pro_cg.pdb' % str(i).zfill(4))
            X_list2, Y_list2, Z_list2, atomtype_list2 = read_pdb('../data/training_data/%s_lig_cg.pdb' % str(j).zfill(4))

            pro = np.transpose(np.array([X_list, Y_list, Z_list, atomtype_list]))
            print(pro.shape)
            lig = np.transpose(([X_list2, Y_list2, Z_list2, atomtype_list2]))
            print(lig.shape)

            # note: n = (i-1)*N + j-1
            pro_list.append(pro)
            lig_list.append(lig)

    return np.array(pro_list), np.array(lig_list)

def convert_flag(x):
    if x > 0:
        return 1
    else:
        return -1

def find_nearest_atoms_YMT(pro, lig, num=2):
    pro_lig= []
    for i in range(len(lig)):
        data = []
        distance = np.sqrt((pro[:,0]-lig[i,0])**2+(pro[:,1]-lig[i,1])**2+(pro[:,2]-lig[i,2])**2)
        # here get the locations of the nearest atoms
        index_nearest_atoms = heapq.nsmallest(num, range(len(distance)), distance.take)
        for j in range(num):
            atom = pro[index_nearest_atoms[j]]
            atom[3] = convert_flag(atom[3])
            data.append(atom)
        # remember to modify ligand's type's flag
        lig[i][3] = convert_flag(lig[i][3])
        data.append(lig[i])
        pro_lig.append(data)
    pro_lig = np.array(pro_lig)
    return pro_lig

def build_KDTree(pro):
    t = KDTree(pro)
    return t

def find_nearest_atoms_KDTree(t, lig, num):
    dist, ind = t.query(lig, k=3)
    get_info = lambda x: t.data[x]

    output = []
    for v_p_nodes in ind:
        temp = list(map(get_info, v_p_nodes))
        for atom in temp:
            # lambda is not used here cause some unexpected errors may happen
            atom[3] = convert_flag(atom[3])
        output.append(temp)
    return output

if __name__ == '__main__':
    pro_list, lig_list = extract_data()
    tree_list = []
    for i in pro_list:
        tree_list.append(build_KDTree(i))

    for i in range(M):
        for j in range(N):
            t = tree_list[i]
            pro = pro_list[i]
            lig = lig_list[j]

            print('\n\n*****************************************\nThe KDTree answer is:\n')
            output = find_nearest_atoms_KDTree(t, lig, 3)
            print(np.array(output))

            print("\n\n*****************************************\nThe YMT's answer is:\n")
            print(find_nearest_atoms_YMT(pro, lig, 3))

'''
Here is an alternative method to use sklearn

# from sklearn.neighbors import KDTree as kt2
# t2 = kt2(d_train)
# dist, ind = t2.query(d_valid, k=3)
# 
# output =[]
# get_info = lambda x: list(t2.data[x])
# 
# for v_p_nodes in ind:
#     temp = list(map(get_info, v_p_nodes))
#     for atom in temp:
#         print(atom[3])
#         # lambda is not used here cause some unexpected errors may happen
#         atom[3] = convert_flag(atom[3])
#     output.append(temp)

'''

