# This is the signal generator for voltage signal used in the experimental RC spoken digit recognition task

import csv
import librosa
import numpy as np
import pandas as pd
import os
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy.special import softmax
import seaborn as sns
from Voice_Inputs import *
import mpl_toolkits.axisartist as AA


def conmat_acc(
        words, _output, VL, num_classification
):
    _con_mat = np.zeros((num_classification, num_classification))
    correct = 0
    for i in range(words):
        _out = _output[sum(VL[:i]):sum(VL[:i + 1]), :]
        q_predicted = np.argmax(softmax(np.mean(_out, axis=0), axis=0))
        _, q_truth = divmod(i, num_classification)
        _con_mat[q_truth, q_predicted] += 1
        if q_predicted == q_truth:
            correct += 1

    return _con_mat, correct


def target_signal_gen(
        words,
        _set,
        num_classification,
        device
):
    VL = []
    Target = np.zeros((0, num_classification))
    file_info = np.zeros((0, 3))

    all_list = pd.read_csv(r'./Data/Voice/Exp/{}/inputs/steps_rec.csv'.format(device), header=None).values

    for j in range(words):
        # Read in the audio file
        p, q = divmod(j, num_classification)
        filepath = _set[p, q]
        set_num = int(filepath[-5])
        subject_num = int(filepath[-9])
        digit_num = int(filepath[-11])

        len_target = all_list[set_num + digit_num*10 + (subject_num-1)*100, 0]

        VL.append(len_target)
        add_target = np.zeros((len_target, num_classification))
        add_target[:, q] = 1
        add_file_info = np.array([[digit_num, subject_num, set_num]])
        Target = np.r_[Target, add_target]
        file_info = np.r_[file_info, add_file_info]

    return VL, Target, file_info


def Voice_SRC_exp(
        Device='TiOx',
        num_classification=10,
        num_node=40,
        direct_transfer=False,
        identical=False,
        down_sampling_start=15,
        OUTPUT=False,
        suffix=''
):
    # Device choice
    if Device == 'TiOx':
        device_set = ['4u', '15d', '11u', '16d']
    else:
        device_set = ['Device a', 'Device b', 'Device c', 'Device d']

    if identical:
        print('identical mode')
        Tr_set = device_set[:1]
    elif direct_transfer:  # multi-channel DT
        print('direct transfer')
        Tr_set = device_set[-1:]
    else:  # multi-channel TS
        print('temporal switch')
        Tr_set = device_set[1:]

    filenames = [['' for _ in range(num_classification)] for _ in range(50)]
    for i in range(5):
        for j in range(num_classification):
            for k in range(10):
                filenames[k + i * 10][
                    j] = r'./Data/Voice\Voice Source Data\train\f{}\0{}f{}set{}.wav'.format(
                    i + 1, j, i + 1, k)
    filenames = np.array(filenames)
    S = np.array([np.random.permutation(filenames[:, j]) for j in range(num_classification)]).T

    # 10-fold cross validation
    train_con_mat = []
    test_con_mat = []
    train_acc = []
    test_acc = []
    train_power = []
    test_power = []
    # record power in [mean, max, min, std, N] order
    for u in range(10):

        # Training dataset pre-processing
        words = 45*num_classification
        train_set = np.delete(S, [i for i in range(u * 5, (u + 1) * 5)], axis=0)
        test_set = S[u * 5:(u + 1) * 5, :]

        VL, Target, FileInfo = target_signal_gen(words, train_set, num_classification, device=Device)

        # Masking and scaling to voltage

        X = np.zeros((sum(VL), num_node))
        power_series = np.zeros((sum(VL)*num_node, 1))

        # Training
        if not direct_transfer and not identical:
            switches = 3
            period = 15*num_classification  # 150 words for each switch
        else:
            switches = 1
            period = 45*num_classification

        counter = 0

        for k in range(switches):
            for count in range(period):
                marker1, marker2 = sum(VL[:k*period+count]), sum(VL[:k*period+count+1])
                slice_len = marker2 - marker1
                if Device == 'TiOx':
                    _file = r'.\Data\Voice/Exp\TiOx/response\{}\{}_0{}f{}set{}.csv'.format(
                        Tr_set[k],
                        Tr_set[k],
                        int(FileInfo[k*period+count, 0]),
                        int(FileInfo[k*period+count, 1]),
                        int(FileInfo[k*period+count, 2]))
                else:  # Device == 'NbOx'
                    _file = r'.\Data\Voice/Exp\NbOx/response\{}\0{}f{}set{}.csv'.format(
                        Tr_set[k],
                        int(FileInfo[k*period+count, 0]),
                        int(FileInfo[k*period+count, 1]),
                        int(FileInfo[k*period+count, 2]))

                df = pd.read_csv(_file, header=None, sep='\n')
                df = df[0].str.split(',', expand=True)
                df_0 = df.iloc[148:20148, 1:4]
                df_0_numpy = df_0.to_numpy()
                data = df_0_numpy.astype(np.float64)
                data_RC_one_device = - data[:, 2]
                voltages = data[:, 1]

                down_sampling_ratio = int(len(data_RC_one_device) / 25 / 40)
                # 25 is the max length for one slice, 40 is the num_node
                data_resampled = data_RC_one_device[down_sampling_start::down_sampling_ratio]

                data_eff = data_resampled[:slice_len*40]
                X[marker1:marker2, 0:num_node] = data_eff.reshape(slice_len, num_node)

                counter += 1
                print('Now is the {}-th read-in for {}-th fold.'.format(counter, u))


        # Linear regression
        lin = Ridge(alpha=0)
        lin.fit(X, Target)
        Output_tr = lin.predict(X)

        # confusion matrix for training
        _con_mat, correct = conmat_acc(words, Output_tr, VL, num_classification)
        print('Fold No.{}, training acc = {:2f}%'.format(u+1, correct/words*100))
        print('training power consumption is {} uJ'.format(1e6*np.mean(np.sum(power_series, axis=1))))
        train_con_mat.append(_con_mat)
        train_acc.append(round(correct/words*100, 2))
        train_power.append(power_series)

        # Testing dataset pre-processing
        words = 5*num_classification
        VL, Target, FileInfo = target_signal_gen(words, test_set, num_classification)

        # Masking and scaling to voltage
        X = np.zeros((sum(VL), num_node))
        power_series = np.zeros((sum(VL)*num_node, 1))

        # Testing
        for count in range(words):
            marker1, marker2 = sum(VL[:count]), sum(VL[:count+1])
            slice_len = marker2 - marker1
            if Device == 'TiOx':
                _file = r'.\Data\Voice Exp\response\{}\{}_0{}f{}set{}.csv'.format(
                    device_set[0],
                    device_set[0],
                    int(FileInfo[count, 0]),
                    int(FileInfo[count, 1]),
                    int(FileInfo[count, 2]))
            else:
                _file = r'.\Data\Voice Exp\response\{}\0{}f{}set{}.csv'.format(
                    device_set[0],
                    int(FileInfo[count, 0]),
                    int(FileInfo[count, 1]),
                    int(FileInfo[count, 2]))
            df = pd.read_csv(_file, header=None, sep='\n')
            df = df[0].str.split(',', expand=True)
            df_0 = df.iloc[148:20148, 1:4]
            df_0_numpy = df_0.to_numpy()
            data = df_0_numpy.astype(np.float64)
            data_RC_one_device = - data[:, 2]
            voltages = data[:, 1]

            down_sampling_ratio = int(len(data_RC_one_device) / 25 / 40)
            # 25 is the max length for one slice, 40 is the num_node
            data_resampled = data_RC_one_device[down_sampling_start::down_sampling_ratio]

            data_eff = data_resampled[:slice_len * 40]
            X[marker1:marker2, 0:num_node] = data_eff.reshape(slice_len, num_node)

        # Output
        Output_ts = lin.predict(X)

        # confusion matrix for training
        _con_mat, correct = conmat_acc(words, Output_ts, VL, num_classification)
        print('Fold No.{}, testing acc = {:2f}%'.format(u+1, correct/words*100))
        print('testing power consumption is {} uJ'.format(1e6*np.mean(np.sum(power_series, axis=1))))
        test_con_mat.append(_con_mat)
        test_acc.append(round(correct/words*100, 2))
        test_power.append(power_series)

    if OUTPUT:
        return train_acc, train_con_mat, train_power, test_acc, test_con_mat, test_power
    else:
        # save the data into csv file
        # Confusion matrix of training
        SEPARATOR = ['---END_OF_MATRIX---']
        if identical:
            mode_name = 'ID'
        elif direct_transfer:
            mode_name = 'DT'
        else:
            mode_name = 'TS'

        with open(r"./Data/Voice/Exp/{}/outputs/{}_train_con_mat{}.csv".format(Device, mode_name, suffix), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for matrix in train_con_mat:
                for row in matrix:
                    writer.writerow(row)
                writer.writerow(SEPARATOR)

        # Confusion matrix of testing
        with open(r"./Data/Voice/Exp/{}/outputs/{}_test_con_mat{}.csv".format(Device, mode_name, suffix), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for matrix in test_con_mat:
                for row in matrix:
                    writer.writerow(row)
                writer.writerow(SEPARATOR)

        # Both accuracies
        both_acc = np.array([train_acc, test_acc])
        with open(r"./Data/Voice/Exp/{}/outputs/{}_both_acc{}.csv".format(Device, mode_name, suffix), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for row in both_acc:
                writer.writerow(row)

        return None


def con_mat_plot(filename, num_classification, Device):

    df = pd.read_csv('./Data/Voice/Exp/{}/outputs/'.format(Device)+filename+'.csv', header=None, sep='\n')
    # ATTENTION: sep='\n' only works for pandas v1.3.4
    confusion_matrix = np.zeros((num_classification, num_classification))
    df = df[0].str.split(',', expand=True)
    for i in range(num_classification):
        df_0 = df.iloc[0 + (num_classification+1)*i:num_classification+(num_classification+1)*i, 0:num_classification]
        df_0 = df_0.to_numpy()
        confusion_matrix_one_trial = df_0.astype(np.float64)
        confusion_matrix_one_trial = np.round(confusion_matrix_one_trial, decimals=0).astype(int)
        confusion_matrix += confusion_matrix_one_trial

    plt.figure(figsize=(4, 3.5))
    sns.set(font_scale=0.8)

    ax = sns.heatmap(
        confusion_matrix,
        annot=False,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        linecolor="white",
        cbar=True,  # 显示颜色条
        annot_kws={"size": 12},  # 注释字体大小
    )

    # 设置坐标轴标签
    ax.set_xlabel("Predicted label", fontsize=8)
    ax.set_ylabel("True label", fontsize=8)
    # 设置类别标签（假设类别为 0-9）
    class_labels = [str(i) for i in range(num_classification)]
    ax.set_xticklabels(class_labels, rotation=0)
    ax.set_yticklabels(class_labels, rotation=0)
    plt.tight_layout()

    plt.savefig('./Figure/Voice/Exp/{}/'.format(Device)+filename+'.svg', format='svg', dpi=300,
                bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()

    pass



if __name__ == '__main__':

    Devices = ['TiOx', 'NbOx']  # type in 'NbOx' or 'TiOx'

    for Device in Devices:

        # Check whether the folder for storing figures is created
        fig_dir = './Figure/Voice/Exp/{}'.format(Device)
        data_dir1 = './Data/Voice/Exp/{}/inputs'.format(Device)
        data_dir2 = './Data/Voice/Exp/{}/outputs'.format(Device)
        data_dir3 = './Data/Voice/Exp/{}/response'.format(Device)

        if not os.path.exists(fig_dir):
            print('Creating new figure file directory...')
            os.makedirs(fig_dir)
        if not os.path.exists(data_dir1):
            print('Creating new data file directory...')
            os.makedirs(data_dir1)
        if not os.path.exists(data_dir2):
            print('Creating new data file directory...')
            os.makedirs(data_dir2)
        if not os.path.exists(data_dir3):
            print('Creating new data file directory...')
            os.makedirs(data_dir2)

        num_classification = 10

        # create_Voice_signal_file(data_dir=data_dir1)  # please disable this function when the voice signal file is generated
        # Voice_SRC_exp(direct_transfer=True, num_classification=num_classification, Device=Device)  # DT
        # Voice_SRC_exp(num_classification=num_classification, Device=Device)  # TS

        # Voice_SRC_exp(identical=True, num_classification=num_classification, Device=Device)   # identical

        con_mat_plot('DT_train_con_mat', num_classification=num_classification, Device=Device)
        con_mat_plot('DT_test_con_mat', num_classification=num_classification, Device=Device)
        con_mat_plot('ID_train_con_mat', num_classification=num_classification, Device=Device)
        con_mat_plot('ID_test_con_mat', num_classification=num_classification, Device=Device)
        con_mat_plot('TS_train_con_mat', num_classification=num_classification, Device=Device)
        con_mat_plot('TS_test_con_mat', num_classification=num_classification, Device=Device)

