import numpy as np
import pandas as pd

M = N = 1

def read_pdb(filename):
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

        line_length = len(stripped_line)
        # print("Line length:{}".format(line_length))
        if line_length < 78:
            print("ERROR: line length is different. Expected>=78, current={}".format(line_length))

        X_list.append(float(stripped_line[30:38].strip()))
        Y_list.append(float(stripped_line[38:46].strip()))
        Z_list.append(float(stripped_line[46:54].strip()))

        atomtype = stripped_line[76:78].strip()
        if atomtype == 'C':
            atomtype_list.append(0.001)  # 'h' means hydrophobic
        else:
            atomtype_list.append(-0.001)  # 'p' means polar

    return X_list, Y_list, Z_list, atomtype_list


def extract_data():
    input = []
    output = []
    for i in range(1, M+1):
        for j in range(1, N+1):
            X_list, Y_list, Z_list, atomtype_list = read_pdb('../data/training_data/%s_pro_cg.pdb' % str(i).zfill(4))
            X_list2, Y_list2, Z_list2, atomtype_list2 = read_pdb('../data/training_data/%s_lig_cg.pdb' % str(j).zfill(4))

            pro = np.array([X_list, Y_list, Z_list, atomtype_list])
            # print(pro.shape)
            lig = np.array([X_list2, Y_list2, Z_list2, atomtype_list2])
            # print(lig.shape)

            data = np.concatenate([pro, lig], axis=1)
            data = np.transpose(data)
            # print(data)
            # note: n = (i-1)*N + j-1
            input.append(data)
            output.append(1 if i==j else 0)

    input = np.array(input)
    print(input.shape)

    return sort_data(input[0])


def compute_vec_dist(vec1, vec2):
    return np.linalg.norm(np.array(vec1) - np.array(vec2))


def sort_data(data):
    # get the minimum values of x, y, z as vec_o
    min_v = data.min(0)
    vec_o = np.delete(min_v, 3)
    # sort the data frame by the distance to vec_o, then drop the dist
    df = pd.DataFrame(data, columns=['x', 'y', 'z', 'type'])
    df['dist'] = df.apply(lambda x: compute_vec_dist([x['x'], x['y'], x['z']], vec_o), axis = 1)
    df = df.sort_values(by=['dist']).drop(columns=['dist'])

    print(df.values)
    return df.values


def find_path():
    pass


if __name__ == '__main__':
    extract_data()
    pass

