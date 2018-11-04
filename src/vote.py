import numpy as np


def vote(w1, w2, w3, file1, file2, file3):
    answer = []

    answer1 = np.loadtxt(file1, dtype=np.int, delimiter='\t',
                         skiprows=1)
    answer2 = np.loadtxt(file2, dtype=np.int, delimiter='\t',
                         skiprows=1)
    answer3 = np.loadtxt(file3, dtype=np.int, delimiter='\t',
                         skiprows=1)
    for i in range(824):
        num_dict = {}
        for j in range(1, 11):
            if answer1[i][j] in num_dict:
                num_dict[answer1[i][j]] = num_dict[answer1[i][j]] + 1 * w1
            else:
                num_dict[answer1[i][j]] = 1 * w1

            if answer2[i][j] in num_dict:
                num_dict[answer2[i][j]] = num_dict[answer2[i][j]] + 1 * w2
            else:
                num_dict[answer2[i][j]] = 1 * w2

            if answer3[i][j] in num_dict:
                num_dict[answer3[i][j]] = num_dict[answer3[i][j]] + 1 * w3
            else:
                num_dict[answer3[i][j]] = 1 * w3

        result = sorted(num_dict.items(), key=lambda item: item[1])
        result.reverse()
        subanswer = [i+1]
        for w in range(10):
            subanswer.append(result[w][0])
        answer.append(subanswer)
    answer = np.array(answer)

    return answer

if __name__ == '__main__':
    w1 = 0.6
    w2 = 0.7
    w3 = 0.6
    filename1 = '../data/result/test_predictions_mlp_example.txt'
    filename2 = '../data/result/test_predictions_lstm_example.txt'
    filename3 = '../data/result/test_predictions_cnn_example.txt'
    answer = vote(w1, w2,  filename1, filename2 )
    np.savetxt('../data/result/test_predictions_combined.txt', answer, delimiter='\t', newline='\n', comments='',
               header='pro_id\tlig1_id\tlig2_id\tlig3_id\tlig4_id\tlig5_id\tlig6_id\tlig7_id\tlig8_id\tlig9_id\tlig10_id',
               fmt='%d')








