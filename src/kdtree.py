import numpy as np
import heapq
from scipy.spatial import KDTree

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
    # ind = [[nodes for atom1], [nodes for atom2]]
    dist, ind = t.query(lig, num)
    get_info = lambda x: t.data[x]

    output = []
    for i in range(len(ind)):
        temp = list(map(get_info, ind[i]))
        for atom in temp:
            # lambda is not used here cause some unexpected errors may happen
            atom[3] = convert_flag(atom[3])

        # remember to modify ligand's flag
        lig[i][3] = convert_flag(lig[i][3])
        temp.append(lig[i])
        output.append(temp)

    return np.array(output)


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

