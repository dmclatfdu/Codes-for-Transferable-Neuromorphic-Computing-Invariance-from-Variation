"""
This is the file for the code relevant to the Spoken Digit Recognition task, including prediction and transfer in both
classical and TS framework, RC of three parallel channels, modified from the MATLAB code by Tsinghua-LEMON-Lab
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from sim_RC_library import *
from scipy import signal
from scipy.special import softmax
import librosa


def voice_signal_gen(
        words,
        _set,
        n_mfcc,
        convolution,
        mean_row=None,
        max_row=None,
        min_row=None
):
    VL = []
    Target = np.zeros((0, 10))
    Input = np.zeros((0, n_mfcc))

    for j in range(words):
        # Read in the audio file
        p, q = divmod(j, 10)
        filepath = _set[p, q]
        audio, sr = librosa.load(filepath, sr=None)
        audio = librosa.resample(audio, orig_sr=12500, target_sr=8000)

        # Pre-processing
        # use the MFCC to replace the LyonPassiveEar function, to extract the spectral feature from the audio
        Feature = librosa.feature.mfcc(y=audio, sr=8000, dct_type=2, n_mfcc=n_mfcc).T

        if j != 0 or mean_row is not None or min_row is not None or max_row is not None:
            pass
        # normalize to [-1, 1], given by factors from the first sample
        else:
            max_row = np.max(Feature, axis=0)
            min_row = np.min(Feature, axis=0)
            mean_row = np.mean(Feature, axis=0)

        Feature = (Feature - mean_row) / (max_row - min_row)

        # 1D convolution
        if convolution:
            window = signal.windows.hann(20)
            feature = np.zeros(Feature.shape)
            for i in range(len(Feature[0, :])):
                feature[:, i] = signal.convolve(Feature[:, i], window, mode='same') / sum(window)
        else:
            feature = Feature

        VL.append(int(len(feature[:, 0])))
        add_target = np.zeros((len(feature[:, 0]), 10))
        add_target[:, q] = 1
        Input = np.r_[Input, feature]
        Target = np.r_[Target, add_target]

    return Input, VL, Target, mean_row, max_row, min_row


def conmat_acc(
        words, _output, VL
):
    _con_mat = np.zeros((10, 10))
    correct = 0
    for i in range(words):
        _out = _output[sum(VL[:i]):sum(VL[:i + 1]), :]
        q_predicted = np.argmax(softmax(np.mean(_out, axis=0), axis=0))
        _, q_truth = divmod(i, 10)
        _con_mat[q_truth, q_predicted] += 1
        if q_predicted == q_truth:
            correct += 1

    return _con_mat, correct


def Voice_SRC_sim(
        direct_transfer=False,
        convolution=True,
        identical=False,
        n_mfcc=32,
        num_node=100,
        noise_level=1e-6,
        C2C_tr=0.01e-5, C2C_ts=0.01e-5,
        channels=1,
        OUTPUT=True,
        suffix=''
):
    # multi-channel and direct transfer settings

    if channels == 1:  # single channel DT
        print('single channel')
        Ts_k3 = np.array([[1.16e-5]])
        if identical:
            Tr_set_k3 = Ts_k3
            print('identical mode')
        elif direct_transfer:
            print('direct transfer')
            Tr_set_k3 = np.array([[0.96e-5]])
        else:  # single channel TS
            print('temporal switch')
            Tr_set_k3 = np.array([np.linspace(0.96e-5, 1.2e-5, 3)]).T
    elif channels > 1 and type(channels) is int:
        print('multi-channel')
        Ts_k3 = np.random.uniform(0.96e-5, 1.2e-5, (1, channels))
        if identical:
            print('identical mode')
            Tr_set_k3 = Ts_k3
        elif direct_transfer:  # multi-channel DT
            print('direct transfer')
            Tr_set_k3 = np.random.uniform(0.96e-5, 1.2e-5, (1, channels))
        else:  # multi-channel TS
            print('temporal switch')
            Tr_set_k3 = np.random.uniform(0.96e-5, 1.2e-5, (3, channels))
    else:
        print('Wrong setting of channels')
        return None

    print('k3 for training is {}'.format(Tr_set_k3))
    print('for testing is {}'.format(Ts_k3))

    # create mask
    mask = create_mask(num_node * channels, in_dim=n_mfcc, abs_value=0.1)
    # Create the SRC module
    SRC = TiOx_SRC()

    filenames = [['' for _ in range(10)] for _ in range(50)]
    for i in range(5):
        for j in range(10):
            for k in range(10):
                filenames[k + i * 10][
                    j] = r'./Data/Voice/Voice Source Data\train\f{}\0{}f{}set{}.wav'.format(
                    i + 1, j, i + 1, k)
    filenames = np.array(filenames)
    WRR = 0
    TF = np.zeros((10, 10))
    S = np.array([np.random.permutation(filenames[:, j]) for j in range(10)]).T

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
        words = 450
        train_set = np.delete(S, [i for i in range(u * 5, (u + 1) * 5)], axis=0)
        test_set = S[u * 5:(u + 1) * 5, :]

        Input, VL, Target, mean_row, max_row, min_row = voice_signal_gen(words, train_set, n_mfcc, convolution)

        # Masking and scaling to voltage
        Masked_Input = Input @ mask
        _min, _max = np.min(Masked_Input), np.max(Masked_Input)
        scaling_factors = np.array([_min, _max])
        Scaled_Input = (Masked_Input - (_min + _max) / 2) / (_max - _min) * (2.5 - 2) + (2.5 + 2) / 2
        Channel_Input = np.array(np.split(Scaled_Input, channels, axis=1))

        X = np.zeros((sum(VL), num_node*channels))
        power_series = np.zeros((sum(VL)*num_node, channels))

        # Training
        if not direct_transfer and not identical:
            switches = 3
            period = 150  # 150 words for each switch
        else:
            switches = 1
            period = 450

        for k in range(switches):
            for count in range(period):
                marker1, marker2 = sum(VL[:k*period+count]), sum(VL[:k*period+count+1])
                for i in range(channels):
                    Channel_Input_sl = Channel_Input[i, marker1:marker2, :]
                    i_tr, g_tr, g0_tr = SRC.iterate_SRC(Channel_Input_sl.flatten(), 20e-6, k3=Tr_set_k3[k, i],
                                                        virtual_nodes=num_node,
                                                        clear=True,
                                                        tqdm_on=False,
                                                        C2C_strength=C2C_tr)
                    # Noising
                    i_tr += noise_level*np.random.randn(i_tr.shape[0])
                    power_series[marker1*num_node:marker2*num_node, i] = i_tr * Channel_Input_sl.flatten()
                    X[marker1:marker2, i*num_node:(i+1)*num_node] = i_tr.reshape(Channel_Input_sl.shape)

        # Linear regression
        lin = Ridge(alpha=0)
        lin.fit(X, Target)
        Output_tr = lin.predict(X)

        # confusion matrix for training
        _con_mat, correct = conmat_acc(words, Output_tr, VL)
        print('Fold No.{}, training acc = {:2f}%'.format(u+1, correct/words*100))
        print('training power consumption is {} uJ'.format(1e6*np.mean(np.sum(power_series, axis=1))))
        train_con_mat.append(_con_mat)
        train_acc.append(round(correct/words*100, 2))
        train_power.append(power_series)

        # Testing dataset pre-processing
        words = 50
        Input, VL, Target, *_ = voice_signal_gen(words, test_set, n_mfcc, convolution,
                                                 mean_row=mean_row, max_row=max_row, min_row=min_row)

        # Masking and scaling to voltage
        Masked_Input = Input @ mask
        Scaled_Input = (Masked_Input - (_min + _max) / 2) / (_max - _min) * (2.5 - 2) + (2.5 + 2) / 2
        Channel_Input = np.array(np.split(Scaled_Input, channels, axis=1))
        X = np.zeros((sum(VL), num_node*channels))
        power_series = np.zeros((sum(VL)*num_node, channels))

        # Testing
        for count in range(words):
            marker1, marker2 = sum(VL[:count]), sum(VL[:count+1])
            for i in range(channels):
                Channel_Input_sl = Channel_Input[i, marker1:marker2, :]
                i_ts, g_ts, g0_ts = SRC.iterate_SRC(Channel_Input_sl.flatten(), 20e-6, k3=Ts_k3[0, i],
                                                    virtual_nodes=num_node,
                                                    clear=True,
                                                    tqdm_on=False,
                                                    C2C_strength=min(C2C_ts, 0.01e-5))
                # Noising
                i_ts += noise_level*np.random.randn(i_ts.shape[0])
                power_series[marker1*num_node:marker2*num_node, i] = i_ts * Channel_Input_sl.flatten()
                X[marker1:marker2, i*num_node:(i+1)*num_node] = i_ts.reshape(Channel_Input_sl.shape)

        # Output
        Output_ts = lin.predict(X)

        # confusion matrix for training
        _con_mat, correct = conmat_acc(words, Output_ts, VL)
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

        with open(r"./Data/Voice/Sim/{}_train_con_mat{}.csv".format(mode_name, suffix), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for matrix in train_con_mat:
                for row in matrix:
                    writer.writerow(row)
                writer.writerow(SEPARATOR)

        # Confusion matrix of testing
        with open(r"./Data/Voice/Sim/{}_test_con_mat{}.csv".format(mode_name, suffix), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for matrix in test_con_mat:
                for row in matrix:
                    writer.writerow(row)
                writer.writerow(SEPARATOR)

        # Both accuracies
        both_acc = np.array([train_acc, test_acc])
        with open(r"./Data/Voice/Sim/{}_both_acc{}.csv".format(mode_name, suffix), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for row in both_acc:
                writer.writerow(row)

        return None


def con_mat_plot(filename):

    df = pd.read_csv('./Data/Voice/Sim/'+filename+'.csv', header=None, sep='\n')
    # ATTENTION: sep='\n' only works for pandas v1.3.4
    confusion_matrix = np.zeros((10, 10))
    df = df[0].str.split(',', expand=True)
    for i in range(10):
        df_0 = df.iloc[0 + 11*i:10+11*i, 0:10]
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
    class_labels = [str(i) for i in range(10)]
    ax.set_xticklabels(class_labels, rotation=0)
    ax.set_yticklabels(class_labels, rotation=0)
    plt.tight_layout()

    plt.savefig('./Figure/Voice/Sim/'+filename+'.svg', format='svg', dpi=300,
                bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()

    pass


def STE_compare_plot():

    filename_list = ['DT_both_acc_ste{}.csv'.format(3*i) for i in range(6)]
    filename_list.append('TS_both_acc_ste.csv')

    data_dir = "./Data/Voice/Sim"
    num_levels = 7
    groups = ['Train', 'Test']
    num_elements = 10

    means = {'Train': np.zeros(num_levels), 'Test': np.zeros(num_levels)}
    errors = {'Train': np.zeros(num_levels), 'Test': np.zeros(num_levels)}

    for level in range(num_levels):

        file_path = os.path.join(data_dir, filename_list[level])

        df = pd.read_csv(file_path, header=None)

        data_A = df.iloc[0, :].values.astype(float)
        data_B = df.iloc[1, :].values.astype(float)

        means['Train'][level] = np.mean(data_A)
        errors['Train'][level] = np.std(data_A)
        means['Test'][level] = np.mean(data_B)
        errors['Test'][level] = np.std(data_B)

    x_labels = ['STE\n$\sigma_{C2C}$=' + '{}'.format(np.round(0.03*i, decimals=2)) for i in range(6)]
    x_labels.append('TS\n$\sigma_{C2C}$=0')
    x = np.arange(num_levels)
    bar_width = 0.35

    light_colors = ['#90CAF9', '#81C784']
    dark_colors = ['#1976D2', '#388E3C']
    special_label_color = '#8B0000'

    fig, ax = plt.subplots(figsize=(8, 3.5))
    plt.rcParams['font.family'] = 'Arial'

    for i in range(len(x_labels)):

        if i < len(x_labels) - 1:
            color_A = light_colors[0]
            color_B = light_colors[1]
            error_color = 'dimgray'
        else:
            color_A = dark_colors[0]
            color_B = dark_colors[1]
            error_color = 'black'

        # Train position
        bar_A = ax.bar(
            x[i] - bar_width / 2, means['Train'][i], bar_width,
            color=color_A, edgecolor='none', capsize=5, label='Train' if i == 0 else ''
        )

        bar_B = ax.bar(
            x[i] + bar_width / 2, means['Test'][i], bar_width,
            color=color_B, edgecolor='none', capsize=5, label='Test' if i == 0 else ''
        )

        ax.errorbar(x[i] - bar_width/2, means['Train'][i], yerr=errors['Train'][i],
                    ecolor=error_color, capsize=5, elinewidth=1.5, capthick=1.5, fmt='none'
                    )

        ax.errorbar(x[i] + bar_width/2, means['Test'][i], yerr=errors['Test'][i],
                    ecolor=error_color, capsize=5, elinewidth=1.5, capthick=1.5, fmt='none')

    ax.tick_params(
        axis='both',
        direction='in',
        which='both',
    )
    ax.set_ylabel("Accuracy (%)", fontsize=8, fontfamily='Arial')
    ax.set_ylim([0, 100])
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.tick_params(axis='both', labelsize=8, labelcolor='black')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Arial')
    ax.legend(fontsize=8)
    ax.axvline(x=5.5, color='gray', linestyle='--', alpha=0.7)

    xtick_labels = ax.get_xticklabels()
    for label in xtick_labels:
        if label.get_text() == x_labels[-1]:  # 找到最后一个标签
            label.set_color(special_label_color)  # 设置为暗红色

    plt.tight_layout()
    plt.savefig('./Figure/Voice/Sim/STE_compare.svg', format='svg', dpi=300,
                transparent=True, bbox_inches='tight')
    plt.show()

    pass

if __name__ == '__main__':

    # Check whether the folder for storing figures is created
    fig_dir = './Figure/Voice/Sim'
    data_dir = './Data/Voice/Sim'
    if not os.path.exists(fig_dir):
        print('Creating new figure file directory...')
        os.makedirs(fig_dir)
    if not os.path.exists(data_dir):
        print('Creating new data file directory...')
        os.makedirs(data_dir)

    # The most accurate baseline is identical=True, convolution=False, C2C_variation=0, noise_level=0

    # # Baseline under C2C variation and electrical noises
    Voice_SRC_sim(identical=True, channels=1, num_node=200, convolution=False, OUTPUT=False)
    # # TS framework
    Voice_SRC_sim(direct_transfer=False, channels=1, num_node=200, convolution=False, OUTPUT=False)
    # # Direct transfer
    Voice_SRC_sim(direct_transfer=True, channels=1, num_node=200, convolution=False, OUTPUT=False)

    # # STE method
    Voice_SRC_sim(direct_transfer=True, channels=1, num_node=200, convolution=False, OUTPUT=False,
                  C2C_tr=0.15e-5, C2C_ts=0, suffix='_ste15')
    Voice_SRC_sim(direct_transfer=True, channels=1, num_node=200, convolution=False, OUTPUT=False,
                  C2C_tr=0.12e-5, C2C_ts=0, suffix='_ste12')
    Voice_SRC_sim(direct_transfer=True, channels=1, num_node=200, convolution=False, OUTPUT=False,
                  C2C_tr=0.09e-5, C2C_ts=0, suffix='_ste9')
    Voice_SRC_sim(direct_transfer=True, channels=1, num_node=200, convolution=False, OUTPUT=False,
                  C2C_tr=0.06e-5, C2C_ts=0, suffix='_ste6')
    Voice_SRC_sim(direct_transfer=True, channels=1, num_node=200, convolution=False, OUTPUT=False,
                  C2C_tr=0.03e-5, C2C_ts=0, suffix='_ste3')
    Voice_SRC_sim(direct_transfer=True, channels=1, num_node=200, convolution=False, OUTPUT=False,
                  C2C_tr=0, C2C_ts=0, suffix='_ste0')
    # # TS framework without C2C, the suffix '_ste' suggests it is tested for the comparison with STE method
    Voice_SRC_sim(direct_transfer=False, channels=1, num_node=200, convolution=False, OUTPUT=False,
                  C2C_tr=0, C2C_ts=0, suffix='_ste')


    STE_compare_plot()

    for i in range(6):
        con_mat_plot('DT_test_con_mat_ste{}'.format(i*3))

    con_mat_plot('TS_test_con_mat_ste')











