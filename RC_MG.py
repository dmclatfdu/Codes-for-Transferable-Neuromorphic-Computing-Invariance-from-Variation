
"""
This is the file for the code relevant to the Mackey-Glass prediction task, including prediction and transfer in both
classical and TS framework, RC of three parallel channels, the relation of NRMSE to device difference and ExtendFig.1
"""

from sim_RC_library import *


def MG_SRC_sim(
        length=1440, shift=1,  # Default MG series settings
        num_node=10, mask_abs=0.1,  # Default RC settings
        direct_transfer=0,
        no_pic=False,
        self=False,
        noise_level=1e-6,
        C2C_variation=0.01e-5,
        num_res=3,  # Number of different reservoirs when direct_transfer is False
        Ts_k3=1.16e-5
):
    # In the training phase, the signal for different memristor reservoir is the same
    MG_gen = MG_generator(0.2, 0.1, 10, 18, shift=shift)
    signal, target = MG_gen.iterate(1, length)
    mask = create_mask(num_node, abs_value=mask_abs)

    Input_tr, input_ts, Target_tr, target_ts = \
        signal_process(signal, target, mask)

    if not direct_transfer:
        input_tr = Input_tr[-int(len(Input_tr) / num_res):]
        target_tr = Target_tr[-int(len(Target_tr) / num_res):]
        target_tr = np.tile(target_tr, (num_res, 1))
    else:
        input_tr = Input_tr
        target_tr = Target_tr
        num_res = 1

    # Create the SRC module
    SRC = TiOx_SRC()

    # Training
    Tr_set_k3 = np.linspace(0.96e-5, 1.2e-5, num_res)

    # # Temporary term
    if self is True:
        Tr_set_k3 = np.linspace(Ts_k3, 1.2e-5, 1)

    State_tr = np.zeros((int(len(Target_tr) / num_res) * num_res, num_node))
    for i in range(num_res):
        i_tr, g_tr, g0_tr = SRC.iterate_SRC(input_tr, 20e-6, k3=Tr_set_k3[i], virtual_nodes=num_node, clear=True,
                                            C2C_strength=C2C_variation)
        State_tr[i * int(len(Target_tr) / num_res):(i + 1) * int(len(Target_tr) / num_res), :] = \
            i_tr.reshape(int(len(Target_tr) / num_res), num_node)

    # Add noise
    State_tr += noise_level * np.random.randn(State_tr.shape[0], State_tr.shape[1])

    # Linear regression
    lin = Ridge(alpha=0)
    lin.fit(State_tr, target_tr)
    Output_tr = lin.predict(State_tr)

    # Testing
    i_ts, g_ts, g0_ts = SRC.iterate_SRC(input_ts, 20e-6, k3=Ts_k3, virtual_nodes=num_node, clear=True,
                                        C2C_strength=C2C_variation)
    State_ts = i_ts.reshape(len(target_ts), num_node)
    State_ts += noise_level * np.random.randn(State_ts.shape[0], State_ts.shape[1])
    Output_ts = lin.predict(State_ts)

    NRMSE_tr, NRMSE_ts = nrmse(target_tr, Output_tr), nrmse(target_ts, Output_ts)

    if not no_pic:
        figure, ax = plt.subplots(2, 2, figsize=(2.4, 2), sharey='row', sharex='col')

        ax1, ax2, ax3, ax4 = ax[1, 0], ax[1, 1], ax[0, 0], ax[0, 1]
        plt.rc('font', family='Arial', size=6)
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['lines.linewidth'] = 1.2

        ylim_max = 0.035

        # Subplot1

        if direct_transfer:
            ax1.plot((Output_tr[:, 0] - target_tr[:, 0]) ** 2, label='Training Error',
                     color=np.array([247, 183, 5]) / 255)
            ax3.plot(target_tr[:, 0], color=np.array([200, 200, 200]) / 255)
            ax3.plot(Output_tr[:, 0], color=np.array([247, 183, 5]) / 255)

        else:
            colors = [np.array([247, 183, 5]) / 255, np.array([255, 97, 101]) / 255, np.array([65, 176, 243]) / 255]
            for i in range(num_res):
                color = colors[i % num_res]
                ax1.axvline(240, ls='--', color=np.array([180, 180, 180]) / 255)
                ax1.axvline(480, ls='--', color=np.array([180, 180, 180]) / 255)
                ax3.axvline(240, ls='--', color=np.array([180, 180, 180]) / 255)
                ax3.axvline(480, ls='--', color=np.array([180, 180, 180]) / 255)
                ax1.plot(np.arange(i * int(len(Target_tr) / num_res), (i + 1) * int(len(Target_tr) / num_res)),
                         (Output_tr[i * int(len(Target_tr) / num_res):(i + 1) * int(len(Target_tr) / num_res), 0] -
                          target_tr[i * int(len(Target_tr) / num_res):(i + 1) * int(len(Target_tr) / num_res), 0]) ** 2,
                         label='Training Error',
                         color=color)
                ax3.plot(
                    np.arange(i * int(len(Target_tr) / num_res), (i + 1) * int(len(Target_tr) / num_res)),
                    target_tr[i * int(len(Target_tr) / num_res):(i + 1) * int(len(Target_tr) / num_res), 0],
                    color=np.array([200, 200, 200]) / 255
                )
                ax3.plot(
                    np.arange(i * int(len(Target_tr) / num_res), (i + 1) * int(len(Target_tr) / num_res)),
                    Output_tr[i * int(len(Target_tr) / num_res):(i + 1) * int(len(Target_tr) / num_res), 0],
                    color=color
                )

        ax1.set_xlim(0, 720)
        ax1.set_ylim(0, ylim_max)
        ax3.set_ylim(0.2, 1.6)

        ax1.set_xlabel('Time step', fontdict={'family': 'arial', 'size': 6}, labelpad=1)
        ax1.set_ylabel('Squared error', fontdict={'family': 'arial', 'size': 6})
        ax3.set_ylabel(r'$x$', fontdict={'family': 'arial', 'size': 6})
        ax1.tick_params(axis='both', direction='in', labelsize=6)
        ax3.tick_params(axis='both', direction='in', labelsize=6)
        ax1.set_xticks([0, 240, 480, 720])
        ax1.set_yticks([0, 0.01, 0.02, 0.03])

        # Subplot2

        ax2.plot(np.arange(720, 1440), (Output_ts[:, 0] - target_ts[:, 0]) ** 2, label='Testing Error',
                 color=np.array([103, 149, 216]) / 255)
        ax4.plot(np.arange(720, 1440), target_ts[:, 0], color=np.array([200, 200, 200]) / 255)
        ax4.plot(np.arange(720, 1440), Output_ts[:, 0], color=np.array([103, 149, 216]) / 255)
        ax2.set_xlim(720, 1440)
        ax2.set_ylim(0, ylim_max)

        # ax2.set_xlabel('Time Step', fontdict={'family': 'arial', 'size': 6})
        ax2.set_xticks([960, 1200, 1440])
        ax2.tick_params(axis='both', direction='in', labelsize=6)
        ax4.tick_params(axis='both', direction='in', labelsize=6)
        figure.subplots_adjust(wspace=0, hspace=0.1)
        plt.show()

        print('NRMSE tr is {}'.format(NRMSE_tr))
        print('NRMSE ts is {}'.format(NRMSE_ts))
    else:
        return NRMSE_tr, NRMSE_ts


def MG_SRC_Expr(
        file_num=20,
        each_length=80,
        dt=1,
        overlap=20,
        out_dim=10,
        warm_up=None,
        tau=None,
        initial=None,
        a=None,
        b=None,
        c=None,
        # Parameters above must be the same with that during the generation of the signal & data
        pred_shift=None,
        num_parallel=1,
        Split=0.5,
        down_sampling_start=15,
        tr_warmup_overlap=None,
        no_pic=False,
        direct_transfer=False,
        choice=3,  # Choosing the mask
        test_device=7  # This 7 is a serial number of device

):
    device_code = ['4u', '6u', '7u', '8u', '9u', '11u', '14d', '15d', '16d']

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% The Generation of Target Series %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Read the file and train and test based on the result from the physical experiment

    point_number = each_length * file_num + overlap

    # Default Settings
    if a is None:
        a = 0.2
    if b is None:
        b = 0.1
    if c is None:
        c = 10
    if tau is None:
        tau = 18

    # Default Setting about the warmup-overlap in actual training-inference of the RC based on physical measurement
    if tr_warmup_overlap is None:
        tr_warmup_overlap = 0
        # tr_warmup_overlap should not be bigger than overlap

    # Points in simulation
    if warm_up is None:  # points for merely warmup, default to be identical length of the needed length
        warm_up = max(int(1 * point_number), 1000)
    if pred_shift is None:
        # pred_shift = 1  # Points for direct prediction shift
        pred_shift = 1
    time_total = (pred_shift + point_number + warm_up) * dt

    x_initial_len = int(tau / dt)
    running_len = (pred_shift + point_number + warm_up)
    x_total_len = x_initial_len + running_len

    # Initialization for MG DDE
    t = np.array([np.arange(0, time_total, dt)]).T
    x_record = np.zeros((x_total_len, 1)) + 0.01
    if initial is None:
        pass
    else:
        x_record[:int(tau / dt), 0] = initial

    for i in range(running_len):
        x_record[i + x_initial_len, 0] = mackey_glass_func(dt, x_record[i + x_initial_len - 1], x_record[i], a, b, c)

    # x_base = x_record[x_initial_len+warm_up:-pred_shift]
    x_target = x_record[x_initial_len + warm_up + pred_shift:]

    # Generate the true target
    x_target = x_target[tr_warmup_overlap:-overlap + tr_warmup_overlap]

    x_target_tr = x_target[:int(len(x_target) * Split)]
    x_target_ts = x_target[int(len(x_target) * Split):]

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% End of the Generation of the Target Series %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Collecting RC Response from the Devices %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Generating the choice of parallel reservoirs

    RC_tr_storage = np.zeros((720, num_parallel * out_dim))
    RC_ts_storage = np.zeros((720, num_parallel * out_dim))
    tr_segments = 9
    ts_segments = 9

    if not direct_transfer:
        train_device_order = [5, 5, 5, 0, 0, 0, 8, 8, 8]
        target_tr = np.tile(x_target_tr[-240:], (3, 1))
        target_ts = x_target_ts[:720]
        train_serial_order = [7, 8, 9, 7, 8, 9, 7, 8, 9]
        test_serial_order = [i+10 for i in range(9)]
    else:
        train_device_order = [8 for i in range(9)]
        target_tr = x_target_tr[-720:]
        target_ts = x_target_ts[:720]
        train_serial_order = [1+i for i in range(9)]
        test_serial_order = [i+10 for i in range(9)]

    for i_tr in range(tr_segments):

        mask_choice = './RC/5um_mask{}'.format(choice)
        file = mask_choice + '/results/5um_{}/5um_{}_Mask{}Seg{}.csv'.format(
            device_code[train_device_order[i_tr]], device_code[train_device_order[i_tr]],
            choice, train_serial_order[i_tr]
        )

        df = pd.read_csv(file, header=None, sep='\n')
        df = df[0].str.split(',', expand=True)
        df_0 = df.iloc[148:20148, 1:4]
        df_0_numpy = df_0.to_numpy()
        data = df_0_numpy.astype(np.float64)
        data_RC_one_device = - data[:, 2]

        # Take the sampling points
        down_sampling_ratio = int(len(data_RC_one_device) / (each_length + overlap) / out_dim)
        data_resampled = data_RC_one_device[down_sampling_start::down_sampling_ratio]
        # Make sure that down_sampling_start should be smaller than down_sampling_ratio
        data_reshaped = np.reshape(data_resampled, (each_length + overlap, out_dim))
        data_response = data_reshaped[tr_warmup_overlap:-overlap + tr_warmup_overlap]

        RC_tr_storage[i_tr * each_length:(i_tr + 1) * each_length, :] = \
            data_response[:, :]
    print('Devices used in training are {}'.format(set(train_device_order)))
    print('Testing device choice is {}'.format(test_device))

    for i_ts in range(ts_segments):
        mask_choice = './RC/5um_mask{}'.format(choice)
        file = mask_choice + '/results/5um_{}/5um_{}_Mask{}Seg{}.csv'.format(
            device_code[test_device], device_code[test_device],
            choice, test_serial_order[i_ts]
        )

        df = pd.read_csv(file, header=None, sep='\n')
        df = df[0].str.split(',', expand=True)
        df_0 = df.iloc[148:20148, 1:4]
        df_0_numpy = df_0.to_numpy()
        data = df_0_numpy.astype(np.float64)
        data_RC_one_device = - data[:, 2]

        # Take the sampling points
        down_sampling_ratio = int(len(data_RC_one_device) / (each_length + overlap) / out_dim)
        data_resampled = data_RC_one_device[down_sampling_start::down_sampling_ratio]
        # Make sure that down_sampling_start should be smaller than down_sampling_ratio
        data_reshaped = np.reshape(data_resampled, (each_length + overlap, out_dim))
        data_response = data_reshaped[tr_warmup_overlap:-overlap + tr_warmup_overlap]

        RC_ts_storage[i_ts * each_length:(i_ts + 1) * each_length, :] = \
            data_response[:, :]

    # RC training
    ridge_alpha = 0
    lin = Ridge(alpha=ridge_alpha)

    lin.fit(RC_tr_storage, target_tr)

    output_tr = lin.predict(RC_tr_storage)
    output_ts = lin.predict(RC_ts_storage)

    NRMSE_tr, NRMSE_ts = nrmse(target_tr, output_tr), nrmse(target_ts, output_ts)
    print('Train NRMSE is {}'.format(NRMSE_tr))
    print('Test NRMSE is {}'.format(NRMSE_ts))

    if not no_pic:

        figure, ax = plt.subplots(2, 2, figsize=(2.4, 2), sharey='row', sharex='col')

        ax1, ax2, ax3, ax4 = ax[1, 0], ax[1, 1], ax[0, 0], ax[0, 1]
        plt.rc('font', family='Arial', size=6)
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['lines.linewidth'] = 1.2

        ylim_max = 0.035

        if direct_transfer:
            ax1.plot((output_tr[:, 0] - target_tr[:, 0]) ** 2, label='Training Error', color=np.array([247, 183, 5]) / 255)
            ax3.plot(target_tr[:, 0], color=np.array([200, 200, 200]) / 255)
            ax3.plot(output_tr[:, 0], color=np.array([247, 183, 5]) / 255)

        else:
            colors = [np.array([247, 183, 5]) / 255, np.array([255, 97, 101]) / 255, np.array([65, 176, 243]) / 255]
            for i in range(3):
                color = colors[i % 3]
                ax1.axvline(240, ls='--', color=np.array([180, 180, 180]) / 255)
                ax1.axvline(480, ls='--', color=np.array([180, 180, 180]) / 255)
                ax3.axvline(240, ls='--', color=np.array([180, 180, 180]) / 255)
                ax3.axvline(480, ls='--', color=np.array([180, 180, 180]) / 255)
                ax1.plot(np.arange(240 * i, 240 * (i+1)),
                         (output_tr[i * 240:(i + 1) * 240, 0] -
                          target_tr[i * 240:(i + 1) * 240, 0]) ** 2,
                         label='Training Error',
                         color=color)
                ax3.plot(
                    np.arange(i * 240, (i + 1) * 240),
                    target_tr[i * 240:(i + 1) * 240, 0],
                    color=np.array([200, 200, 200]) / 255
                )
                ax3.plot(
                    np.arange(i * 240, (i + 1) * 240),
                    output_tr[i * 240:(i + 1) * 240, 0],
                    color=color
                )

        ax1.set_xlim(0, 720)
        ax1.set_ylim(0, ylim_max)
        ax3.set_ylim(0.2, 1.6)

        ax1.set_xlabel('Time step', fontdict={'family': 'arial', 'size': 6}, labelpad=1)
        ax1.set_ylabel('Squared error', fontdict={'family': 'arial', 'size': 6})
        ax3.set_ylabel(r'$x$', fontdict={'family': 'arial', 'size': 6})
        ax1.tick_params(axis='both', direction='in', labelsize=6)
        ax3.tick_params(axis='both', direction='in', labelsize=6)
        ax1.set_xticks([0, 240, 480, 720])
        ax1.set_yticks([0, 0.01, 0.02, 0.03])

        # Subplot2

        ax2.plot(np.arange(720, 1440), (output_ts[:, 0] - target_ts[:, 0]) ** 2, label='Testing Error',
                 color=np.array([103, 149, 216]) / 255)
        ax4.plot(np.arange(720, 1440), target_ts[:, 0], color=np.array([200, 200, 200]) / 255)
        ax4.plot(np.arange(720, 1440), output_ts[:, 0], color=np.array([103, 149, 216]) / 255)
        ax2.set_xlim(720, 1440)
        ax2.set_ylim(0, ylim_max)

        # ax2.set_xlabel('Time Step', fontdict={'family': 'arial', 'size': 6})
        ax2.set_xticks([960, 1200, 1440])
        ax2.tick_params(axis='both', direction='in', labelsize=6)
        ax4.tick_params(axis='both', direction='in', labelsize=6)
        figure.subplots_adjust(wspace=0, hspace=0.1)
        plt.show()

    else:
        return NRMSE_tr, NRMSE_ts


def NRMSE_sim(**kwargs):
    levels = 9
    k3_list = np.linspace(1, 1.25, levels)
    repeat = kwargs.get('repeat', 50)
    dict_nrmse_dt = {}
    dict_nrmse_src = {}

    Storage_nrmse_dt = np.zeros((levels, repeat))
    Storage_nrmse_src = np.zeros((levels, repeat))

    for i in range(levels):
        for j in range(repeat):
            _, Storage_nrmse_dt[i, j] = MG_SRC_sim(
                direct_transfer=True, Ts_k3=0.96e-5 * k3_list[i], no_pic=True
            )
        dict_nrmse_dt['{}'.format(round(100 * (k3_list[i] - 1), 1))] = Storage_nrmse_dt[i, :]

    for i in range(levels):
        for j in range(repeat):
            _, Storage_nrmse_src[i, j] = MG_SRC_sim(
                direct_transfer=False, Ts_k3=0.96e-5 * k3_list[i], no_pic=True
            )
        dict_nrmse_src['{}'.format(round(100 * (k3_list[i] - 1), 1))] = Storage_nrmse_src[i, :]

    with open('./storage_MG_nrmse_classical.csv', mode='w',
              encoding='UTF-8', newline='') as f_tr:
        writer = csv.writer(f_tr)
        writer.writerows(Storage_nrmse_dt)

    with open('./storage_MG_nrmse_TS.csv', mode='w',
              encoding='UTF-8', newline='') as f_ts:
        writer = csv.writer(f_ts)
        writer.writerows(Storage_nrmse_src)

    pass


def NRMSE_sim_plot():
    levels = 9
    k3_list = np.linspace(1, 1.25, levels)
    repeat = 50
    Storage_nrmse_dt = pd.read_csv('./storage_MG_nrmse_classical.csv', header=None).values
    Storage_nrmse_src = pd.read_csv('./storage_MG_nrmse_TS.csv', header=None).values

    dict_nrmse_dt = {}
    dict_nrmse_src = {}

    for i in range(levels):
        dict_nrmse_dt['{}'.format(round(100 * (k3_list[i] - 1), 1))] = Storage_nrmse_dt[i, :]
        dict_nrmse_src['{}'.format(round(100 * (k3_list[i] - 1), 1))] = Storage_nrmse_src[i, :]

    plt.figure(figsize=(2.4, 1.6))
    plt.rc('font', family='Arial', size=6)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1
    sns.boxplot(data=pd.DataFrame(dict_nrmse_dt).iloc[:, :], color=np.array([247, 183, 5]) / 255, fliersize=1.5,
                saturation=1, width=0.7,
                boxprops={'linewidth': 0.5}, whiskerprops={'linewidth': 0.5}, medianprops={'linewidth': 0.5},
                capprops={'linewidth': 0.5}, flierprops={'marker': 'o'})
    sns.pointplot(data=pd.DataFrame(dict_nrmse_dt).iloc[:, :], scale=0.75, color=np.array([247, 183, 5]) / 255,
                  label=r'Classical framework $({\bf W}^{(9)}_{\rm out})$')
    # sns.lineplot(x=k3_list*1.08, y=np.average(Storage_nrmse_dt, axis=1))
    sns.boxplot(data=pd.DataFrame(dict_nrmse_src).iloc[:, :], color=np.array([65, 176, 243]) / 255, fliersize=1.5,
                saturation=1, width=0.7,
                boxprops={'linewidth': 0.5}, whiskerprops={'linewidth': 0.5}, medianprops={'linewidth': 0.5},
                capprops={'linewidth': 0.5}, flierprops={'marker': 'o'})
    sns.pointplot(data=pd.DataFrame(dict_nrmse_src).iloc[:, :], scale=0.75, color=np.array([65, 176, 243]) / 255,
                  label=r'TS training $({\bf W}^{(1,6,9)}_{\rm out})$')
    plt.ylim([0, 0.6])
    plt.legend(frameon=False, loc=2)
    plt.ylabel('NRMSE')
    plt.xlabel('Percentage of difference in k3 (%)')
    plt.show()


def NRMSE_sim(**kwargs):
    levels = 9
    k3_list = np.linspace(1, 1.25, levels)
    repeat = kwargs.get('repeat', 50)
    dict_nrmse_dt = {}
    dict_nrmse_src = {}

    Storage_nrmse_dt = np.zeros((levels, repeat))
    Storage_nrmse_src = np.zeros((levels, repeat))

    for i in range(levels):
        for j in range(repeat):
            _, Storage_nrmse_dt[i, j] = MG_SRC_sim(
                direct_transfer=True, Ts_k3=0.96e-5 * k3_list[i], no_pic=True
            )
        dict_nrmse_dt['{}'.format(round(100 * (k3_list[i] - 1), 1))] = Storage_nrmse_dt[i, :]

    for i in range(levels):
        for j in range(repeat):
            _, Storage_nrmse_src[i, j] = MG_SRC_sim(
                direct_transfer=False, Ts_k3=0.96e-5 * k3_list[i], no_pic=True
            )
        dict_nrmse_src['{}'.format(round(100 * (k3_list[i] - 1), 1))] = Storage_nrmse_src[i, :]

    with open('./storage_MG_nrmse_classical.csv', mode='w',
              encoding='UTF-8', newline='') as f_tr:
        writer = csv.writer(f_tr)
        writer.writerows(Storage_nrmse_dt)

    with open('./storage_MG_nrmse_TS.csv', mode='w',
              encoding='UTF-8', newline='') as f_ts:
        writer = csv.writer(f_ts)
        writer.writerows(Storage_nrmse_src)

    pass


def NRMSE_sim_plot():
    levels = 9
    k3_list = np.linspace(1, 1.25, levels)
    Storage_nrmse_dt = pd.read_csv('./storage_MG_nrmse_classical.csv', header=None).values
    Storage_nrmse_src = pd.read_csv('./storage_MG_nrmse_TS.csv', header=None).values

    dict_nrmse_dt = {}
    dict_nrmse_src = {}

    for i in range(levels):
        dict_nrmse_dt['{}'.format(round(100 * (k3_list[i] - 1), 1))] = Storage_nrmse_dt[i, :]
        dict_nrmse_src['{}'.format(round(100 * (k3_list[i] - 1), 1))] = Storage_nrmse_src[i, :]

    plt.figure(figsize=(2.4, 1.6))
    plt.rc('font', family='Arial', size=6)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1
    sns.boxplot(data=pd.DataFrame(dict_nrmse_dt).iloc[:, :], color=np.array([247, 183, 5]) / 255, fliersize=1.5,
                saturation=1, width=0.7,
                boxprops={'linewidth': 0.5}, whiskerprops={'linewidth': 0.5}, medianprops={'linewidth': 0.5},
                capprops={'linewidth': 0.5}, flierprops={'marker': 'o'})
    sns.pointplot(data=pd.DataFrame(dict_nrmse_dt).iloc[:, :], scale=0.75, color=np.array([247, 183, 5]) / 255,
                  label=r'Classical framework $({\bf W}^{(9)}_{\rm out})$')
    sns.boxplot(data=pd.DataFrame(dict_nrmse_src).iloc[:, :], color=np.array([65, 176, 243]) / 255, fliersize=1.5,
                saturation=1, width=0.7,
                boxprops={'linewidth': 0.5}, whiskerprops={'linewidth': 0.5}, medianprops={'linewidth': 0.5},
                capprops={'linewidth': 0.5}, flierprops={'marker': 'o'})
    sns.pointplot(data=pd.DataFrame(dict_nrmse_src).iloc[:, :], scale=0.75, color=np.array([65, 176, 243]) / 255,
                  label=r'TS training $({\bf W}^{(1,6,9)}_{\rm out})$')
    plt.ylim([0, 0.6])
    plt.legend(frameon=False, loc=2)
    plt.ylabel('NRMSE')
    plt.xlabel('Percentage of difference in k3 (%)')
    plt.show()


def NRMSE_simNoise(**kwargs):
    levels = 8
    noise_list = 1e-6 * np.array([0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5])
    repeat = kwargs.get('repeat', 50)
    dict_nrmse_dt = {}
    dict_nrmse_src = {}
    dict_nrmse_self = {}

    Storage_nrmse_dt = np.zeros((levels, repeat))
    Storage_nrmse_src = np.zeros((levels, repeat))
    Storage_nrmse_self = np.zeros((levels, repeat))

    for i in range(levels):
        for j in range(repeat):
            _, Storage_nrmse_dt[i, j] = MG_SRC_sim(noise_level=noise_list[i],
                                                   direct_transfer=True, Ts_k3=1.16e-5, no_pic=True
                                                   )
        dict_nrmse_dt['{}'.format(round(noise_list[i]*1e6, 1))] = Storage_nrmse_dt[i, :]

    for i in range(levels):
        for j in range(repeat):
            _, Storage_nrmse_src[i, j] = MG_SRC_sim(noise_level=noise_list[i],
                                                    direct_transfer=False, Ts_k3=1.16e-5, no_pic=True
                                                    )
        dict_nrmse_src['{}'.format(round(noise_list[i]*1e6, 1))] = Storage_nrmse_src[i, :]

    for i in range(levels):
        for j in range(repeat):
            _, Storage_nrmse_self[i, j] = MG_SRC_sim(noise_level=noise_list[i], self=True,
                                                     direct_transfer=True, Ts_k3=1.16e-5, no_pic=True
                                                     )
        dict_nrmse_self['{}'.format(round(noise_list[i]*1e6, 1))] = Storage_nrmse_self[i, :]

    with open('./storage_MG_noise_nrmse_classical.csv', mode='w',
              encoding='UTF-8', newline='') as f_dt:
        writer = csv.writer(f_dt)
        writer.writerows(Storage_nrmse_dt)

    with open('./storage_MG_noise_nrmse_TS.csv', mode='w',
              encoding='UTF-8', newline='') as f_src:
        writer = csv.writer(f_src)
        writer.writerows(Storage_nrmse_src)

    with open('./storage_MG_noise_nrmse_self.csv', mode='w',
              encoding='UTF-8', newline='') as f_self:
        writer = csv.writer(f_self)
        writer.writerows(Storage_nrmse_self)

    pass


def NRMSE_simC2C(**kwargs):
    levels = 9
    C2C_list = 0.005e-5 * np.linspace(1, 9, 9)
    repeat = kwargs.get('repeat', 50)
    dict_nrmse_dt = {}
    dict_nrmse_src = {}
    dict_nrmse_self = {}

    Storage_nrmse_dt = np.zeros((levels, repeat))
    Storage_nrmse_src = np.zeros((levels, repeat))
    Storage_nrmse_self = np.zeros((levels, repeat))

    for i in range(levels):
        for j in range(repeat):
            _, Storage_nrmse_dt[i, j] = MG_SRC_sim(noise_level=1e-6, C2C_variation=C2C_list[i],
                                                   direct_transfer=True, Ts_k3=1.16e-5, no_pic=True
                                                   )
        dict_nrmse_dt['{}'.format(round(C2C_list[i]*1e5, 3))] = Storage_nrmse_dt[i, :]

    for i in range(levels):
        for j in range(repeat):
            _, Storage_nrmse_src[i, j] = MG_SRC_sim(noise_level=1e-6, C2C_variation=C2C_list[i],
                                                    direct_transfer=False, Ts_k3=1.16e-5, no_pic=True
                                                    )
        dict_nrmse_src['{}'.format(round(C2C_list[i]*1e5, 3))] = Storage_nrmse_src[i, :]

    for i in range(levels):
        for j in range(repeat):
            _, Storage_nrmse_self[i, j] = MG_SRC_sim(noise_level=1e-6, self=True, C2C_variation=C2C_list[i],
                                                     direct_transfer=True, Ts_k3=1.16e-5, no_pic=True
                                                     )
        dict_nrmse_self['{}'.format(round(C2C_list[i]*1e5, 3))] = Storage_nrmse_self[i, :]

    with open('./storage_MG_c2c_nrmse_classical.csv', mode='w',
              encoding='UTF-8', newline='') as f_dt:
        writer = csv.writer(f_dt)
        writer.writerows(Storage_nrmse_dt)

    with open('./storage_MG_c2c_nrmse_TS.csv', mode='w',
              encoding='UTF-8', newline='') as f_src:
        writer = csv.writer(f_src)
        writer.writerows(Storage_nrmse_src)

    with open('./storage_MG_c2c_nrmse_self.csv', mode='w',
              encoding='UTF-8', newline='') as f_self:
        writer = csv.writer(f_self)
        writer.writerows(Storage_nrmse_self)

    pass


def NRMSE_simNoisePlot(**kwargs):
    levels = 8
    noise_list = 1e-6 * np.array([0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5])
    dict_nrmse_dt = {}
    dict_nrmse_src = {}
    dict_nrmse_self = {}

    Storage_nrmse_dt = pd.read_csv('./storage_MG_noise_nrmse_classical.csv', header=None).values
    Storage_nrmse_src = pd.read_csv('./storage_MG_noise_nrmse_TS.csv', header=None).values
    Storage_nrmse_self = pd.read_csv('./storage_MG_noise_nrmse_self.csv', header=None).values

    for i in range(levels):
        dict_nrmse_dt['{}'.format(round(noise_list[i]*1e6, 1))] = Storage_nrmse_dt[i, :]

    for i in range(levels):
        dict_nrmse_src['{}'.format(round(noise_list[i]*1e6, 1))] = Storage_nrmse_src[i, :]

    for i in range(levels):
        dict_nrmse_self['{}'.format(round(noise_list[i]*1e6, 1))] = Storage_nrmse_self[i, :]

    plt.figure(figsize=(4.2, 2.8))
    plt.rc('font', family='Arial', size=8)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1
    sns.boxplot(data=pd.DataFrame(dict_nrmse_dt).iloc[:, :], color=np.array([247, 183, 5]) / 255, fliersize=1.5,
                saturation=1, width=0.7,
                boxprops={'linewidth': 0.5}, whiskerprops={'linewidth': 0.5}, medianprops={'linewidth': 0.5},
                capprops={'linewidth': 0.5}, flierprops={'marker': 'o'})
    sns.pointplot(data=pd.DataFrame(dict_nrmse_dt).iloc[:, :], scale=0.75, color=np.array([247, 183, 5]) / 255,
                  label=r'Classical framework $({\bf W}^{(9)}_{\rm out})$')
    sns.boxplot(data=pd.DataFrame(dict_nrmse_src).iloc[:, :], color=np.array([65, 176, 243]) / 255, fliersize=1.5,
                saturation=1, width=0.7,
                boxprops={'linewidth': 0.5}, whiskerprops={'linewidth': 0.5}, medianprops={'linewidth': 0.5},
                capprops={'linewidth': 0.5}, flierprops={'marker': 'o'})
    sns.pointplot(data=pd.DataFrame(dict_nrmse_src).iloc[:, :], scale=0.75, color=np.array([65, 176, 243]) / 255,
                  label=r'TS training $({\bf W}^{(1,6,9)}_{\rm out})$')
    sns.boxplot(data=pd.DataFrame(dict_nrmse_self).iloc[:, :], color=np.array([255, 97, 101]) / 255, fliersize=1.5,
                saturation=1, width=0.7,
                boxprops={'linewidth': 0.5}, whiskerprops={'linewidth': 0.5}, medianprops={'linewidth': 0.5},
                capprops={'linewidth': 0.5}, flierprops={'marker': 'o'})
    sns.pointplot(data=pd.DataFrame(dict_nrmse_self).iloc[:, :], scale=0.75, color=np.array([255, 97, 101]) / 255,
                  label=r'Self training $({\bf W}^{(8)}_{\rm out})$')
    plt.ylim([0, 0.6])
    plt.legend(frameon=False, loc=2)
    plt.ylabel('NRMSE')
    plt.xlabel(r'Noise level ($\mu A$)')
    plt.show()


def NRMSE_simC2CPlot(**kwargs):
    levels = 9
    C2C_list = 0.005e-5 * np.linspace(1, 9, 9)
    dict_nrmse_dt = {}
    dict_nrmse_src = {}
    dict_nrmse_self = {}

    Storage_nrmse_dt = pd.read_csv('./storage_MG_c2c_nrmse_classical.csv', header=None).values
    Storage_nrmse_src = pd.read_csv('./storage_MG_c2c_nrmse_TS.csv', header=None).values
    Storage_nrmse_self = pd.read_csv('./storage_MG_c2c_nrmse_self.csv', header=None).values

    for i in range(levels):
        dict_nrmse_dt['{}'.format(round(C2C_list[i]*1e5, 3))] = Storage_nrmse_dt[i, :]

    for i in range(levels):
        dict_nrmse_src['{}'.format(round(C2C_list[i]*1e5, 3))] = Storage_nrmse_src[i, :]

    for i in range(levels):
        dict_nrmse_self['{}'.format(round(C2C_list[i]*1e5, 3))] = Storage_nrmse_self[i, :]

    plt.figure(figsize=(4.2, 2.8))
    plt.rc('font', family='Arial', size=8)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1
    sns.boxplot(data=pd.DataFrame(dict_nrmse_dt).iloc[:, :], color=np.array([247, 183, 5]) / 255, fliersize=1.5,
                saturation=1, width=0.7,
                boxprops={'linewidth': 0.5}, whiskerprops={'linewidth': 0.5}, medianprops={'linewidth': 0.5},
                capprops={'linewidth': 0.5}, flierprops={'marker': 'o'})
    sns.pointplot(data=pd.DataFrame(dict_nrmse_dt).iloc[:, :], scale=0.75, color=np.array([247, 183, 5]) / 255,
                  label=r'Classical framework $({\bf W}^{(9)}_{\rm out})$')
    sns.boxplot(data=pd.DataFrame(dict_nrmse_src).iloc[:, :], color=np.array([65, 176, 243]) / 255, fliersize=1.5,
                saturation=1, width=0.7,
                boxprops={'linewidth': 0.5}, whiskerprops={'linewidth': 0.5}, medianprops={'linewidth': 0.5},
                capprops={'linewidth': 0.5}, flierprops={'marker': 'o'})
    sns.pointplot(data=pd.DataFrame(dict_nrmse_src).iloc[:, :], scale=0.75, color=np.array([65, 176, 243]) / 255,
                  label=r'TS training $({\bf W}^{(1,6,9)}_{\rm out})$')
    sns.boxplot(data=pd.DataFrame(dict_nrmse_self).iloc[:, :], color=np.array([255, 97, 101]) / 255, fliersize=1.5,
                saturation=1, width=0.7,
                boxprops={'linewidth': 0.5}, whiskerprops={'linewidth': 0.5}, medianprops={'linewidth': 0.5},
                capprops={'linewidth': 0.5}, flierprops={'marker': 'o'})
    sns.pointplot(data=pd.DataFrame(dict_nrmse_self).iloc[:, :], scale=0.75, color=np.array([255, 97, 101]) / 255,
                  label=r'Self training $({\bf W}^{(8)}_{\rm out})$')
    plt.ylim([0, 0.6])
    plt.legend(frameon=False, loc=2)
    plt.ylabel('NRMSE')
    plt.xlabel(r'C2C variation strength')
    plt.show()


def MG_SRC_Expr_MultiChannel(
        direct_transfer=False
):


    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% The Generation of Target Series %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Read the file and train and test based on the result from the physical experiment

    # verify_pair = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Default Settings
    down_sampling_start = 15
    tr_warmup_overlap = None

    each_length = 80
    out_dim = 10
    a = 0.2
    b = 0.1
    c = 10
    tau = 18
    dt = 1
    overlap=20
    Split = 0.5

    # Default Setting about the warmup-overlap in actual training-inference of the RC based on physical measurement
    tr_warmup_overlap = 5
    # tr_warmup_overlap should not be bigger than overlap

    # Points in simulation
    # points for merely warmup, default to be identical length of the needed length
    point_number = 80*20+20
    warm_up = max(int(1 * point_number), 1000)
    pred_shift = 1  # pred_shift = 1  # Points for direct prediction shift

    time_total = (pred_shift + point_number + warm_up) * dt

    x_initial_len = int(tau / dt)
    running_len = (pred_shift + point_number + warm_up)
    x_total_len = x_initial_len + running_len

    # Initialization for MG DDE
    t = np.array([np.arange(0, time_total, dt)]).T
    x_record = np.zeros((x_total_len, 1)) + 0.01

    for i in range(running_len):
        x_record[i + x_initial_len, 0] = mackey_glass_func(dt, x_record[i + x_initial_len - 1], x_record[i], a, b, c)

    x_target = x_record[x_initial_len + warm_up + pred_shift:]

    # Generate the true target
    x_target = x_target[tr_warmup_overlap:-overlap + tr_warmup_overlap]

    x_target_tr = x_target[:int(len(x_target) * Split)]
    x_target_ts = x_target[int(len(x_target) * Split):]

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% End of the Generation of the Target Series %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Collecting RC Response from the Devices %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    num_parallel = 3

    RC_tr_storage = np.zeros((720, num_parallel * 10))
    RC_ts_storage = np.zeros((720, num_parallel * 10))
    tr_segments = 9
    ts_segments = 9

    if not direct_transfer:
        train_device_order = [['14d', '8u', '14d'], ['16d', '7u', '4u'], ['9u', '15d', '11u']] * 3
        train_device_order.sort()
        target_tr = np.tile(x_target_tr[-240:], (3, 1))
        target_ts = x_target_ts[:720]
        train_serial_order = [7, 8, 9, 7, 8, 9, 7, 8, 9]
        test_serial_order = [i+10 for i in range(9)]
    else:
        # train_device_order = [8 for i in range(9)]
        train_device_order = [['14d', '8u', '14d']] * 9
        target_tr = x_target_tr[-720:]
        target_ts = x_target_ts[:720]
        train_serial_order = [1+i for i in range(9)]
        test_serial_order = [i+10 for i in range(9)]
    test_device_order = [['16d', '11u', '11u']] * 9

    for i_tr in range(tr_segments):
        for choice in range(num_parallel):
            mask_choice = './RC/5um_mask{}'.format(choice+1)
            file = mask_choice + '/results/5um_{}/5um_{}_Mask{}Seg{}.csv'.format(
                train_device_order[i_tr][choice], train_device_order[i_tr][choice],
                choice+1, train_serial_order[i_tr]
            )

            df = pd.read_csv(file, header=None, sep='\n')
            df = df[0].str.split(',', expand=True)
            df_0 = df.iloc[148:20148, 1:4]
            df_0_numpy = df_0.to_numpy()
            data = df_0_numpy.astype(np.float64)
            data_RC_one_device = - data[:, 2]

            # Take the sampling points
            down_sampling_ratio = int(len(data_RC_one_device) / (each_length + overlap) / out_dim)
            data_resampled = data_RC_one_device[down_sampling_start::down_sampling_ratio]
            # Make sure that down_sampling_start should be smaller than down_sampling_ratio
            data_reshaped = np.reshape(data_resampled, (each_length + overlap, out_dim))
            data_response = data_reshaped[tr_warmup_overlap:-overlap + tr_warmup_overlap]

            RC_tr_storage[i_tr * each_length:(i_tr + 1) * each_length, choice * out_dim:(choice+1) * out_dim] = \
                data_response[:, :]

    for i_ts in range(ts_segments):
        for choice in range(num_parallel):
            mask_choice = './RC/5um_mask{}'.format(choice+1)
            file = mask_choice + '/results/5um_{}/5um_{}_Mask{}Seg{}.csv'.format(
                test_device_order[i_ts][choice], test_device_order[i_ts][choice],
                choice+1, test_serial_order[i_ts]
            )

            df = pd.read_csv(file, header=None, sep='\n')
            df = df[0].str.split(',', expand=True)
            df_0 = df.iloc[148:20148, 1:4]
            df_0_numpy = df_0.to_numpy()
            data = df_0_numpy.astype(np.float64)
            data_RC_one_device = - data[:, 2]

            # Take the sampling points
            down_sampling_ratio = int(len(data_RC_one_device) / (each_length + overlap) / out_dim)
            data_resampled = data_RC_one_device[down_sampling_start::down_sampling_ratio]
            # Make sure that down_sampling_start should be smaller than down_sampling_ratio
            data_reshaped = np.reshape(data_resampled, (each_length + overlap, out_dim))
            data_response = data_reshaped[tr_warmup_overlap:-overlap + tr_warmup_overlap]

            RC_ts_storage[i_ts * each_length:(i_ts + 1) * each_length, choice * out_dim:(choice+1) * out_dim] = \
                data_response[:, :]

    # RC training
    ridge_alpha = 0
    lin = Ridge(alpha=ridge_alpha)

    lin.fit(RC_tr_storage, target_tr)

    x_bar_tr = lin.predict(RC_tr_storage)
    x_bar_ts = lin.predict(RC_ts_storage)

    print('Train NRMSE is {}'.format(nrmse(target_tr, x_bar_tr)))
    print('Test NRMSE is {}'.format(nrmse(target_ts, x_bar_ts)))

    Target_tr = target_tr
    Output_tr = x_bar_tr
    Output_ts = x_bar_ts

    figure, ax = plt.subplots(2, 2, figsize=(3, 2.4), sharey='row', sharex='col')

    ax1, ax2, ax3, ax4 = ax[1, 0], ax[1, 1], ax[0, 0], ax[0, 1]
    plt.rc('font', family='Arial', size=6)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1.2

    ylim_max = 0.06

    # Subplot1

    if direct_transfer:
        ax1.plot((Output_tr[:, 0] - target_tr[:, 0]) ** 2, label='Training Error',
                 color=np.array([247, 183, 5]) / 255)
        ax3.plot(target_tr[:, 0], color=np.array([200, 200, 200]) / 255)
        ax3.plot(Output_tr[:, 0], color=np.array([247, 183, 5]) / 255)

    else:
        colors = [np.array([247, 183, 5]) / 255, np.array([255, 97, 101]) / 255, np.array([65, 176, 243]) / 255]
        num_res = 3
        for i in range(num_res):
            color = colors[i % num_res]
            ax1.axvline(240, ls='--', color=np.array([180, 180, 180]) / 255)
            ax1.axvline(480, ls='--', color=np.array([180, 180, 180]) / 255)
            ax3.axvline(240, ls='--', color=np.array([180, 180, 180]) / 255)
            ax3.axvline(480, ls='--', color=np.array([180, 180, 180]) / 255)
            ax1.plot(np.arange(i * int(len(Target_tr) / num_res), (i + 1) * int(len(Target_tr) / num_res)),
                     (Output_tr[i * int(len(Target_tr) / num_res):(i + 1) * int(len(Target_tr) / num_res), 0] -
                      target_tr[i * int(len(Target_tr) / num_res):(i + 1) * int(len(Target_tr) / num_res), 0]) ** 2,
                     label='Training Error',
                     color=color)
            ax3.plot(
                np.arange(i * int(len(Target_tr) / num_res), (i + 1) * int(len(Target_tr) / num_res)),
                target_tr[i * int(len(Target_tr) / num_res):(i + 1) * int(len(Target_tr) / num_res), 0],
                color=np.array([200, 200, 200]) / 255
            )
            ax3.plot(
                np.arange(i * int(len(Target_tr) / num_res), (i + 1) * int(len(Target_tr) / num_res)),
                Output_tr[i * int(len(Target_tr) / num_res):(i + 1) * int(len(Target_tr) / num_res), 0],
                color=color
            )

    ax1.set_xlim(0, 720)
    ax1.set_ylim(0, ylim_max)
    ax3.set_ylim(0.2, 1.6)

    ax1.set_xlabel('Time step', fontdict={'family': 'arial', 'size': 6}, labelpad=1)
    ax1.set_ylabel('Squared error', fontdict={'family': 'arial', 'size': 6})
    ax3.set_ylabel(r'$x$', fontdict={'family': 'arial', 'size': 6})
    ax1.tick_params(axis='both', direction='in', labelsize=6)
    ax3.tick_params(axis='both', direction='in', labelsize=6)
    ax1.set_xticks([0, 240, 480, 720])
    ax1.set_yticks([0, 0.02, 0.04, 0.06])

    # Subplot2

    ax2.plot(np.arange(720, 1440), (Output_ts[:, 0] - target_ts[:, 0]) ** 2, label='Testing Error',
             color=np.array([103, 149, 216]) / 255)
    ax4.plot(np.arange(720, 1440), target_ts[:, 0], color=np.array([200, 200, 200]) / 255)
    ax4.plot(np.arange(720, 1440), Output_ts[:, 0], color=np.array([103, 149, 216]) / 255)
    ax2.set_xlim(720, 1440)
    ax2.set_ylim(0, ylim_max)

    ax2.set_xticks([960, 1200, 1440])
    ax2.tick_params(axis='both', direction='in', labelsize=6)
    ax4.tick_params(axis='both', direction='in', labelsize=6)
    figure.subplots_adjust(wspace=0, hspace=0.1)
    plt.show()


def NRMSE_expr():

    dict_nrmse_dt = {}
    dict_nrmse_src = {}

    nrmse_record_dt= np.zeros(9)
    nrmse_record_src = np.zeros(9)

    for i in range(9):
        _, nrmse_record_dt[i] = MG_SRC_Expr(
            tr_warmup_overlap=5, pred_shift=1, no_pic=True, direct_transfer=True, test_device=i
        )

        dict_nrmse_dt['{}'.format(i+1)] = nrmse_record_dt[i]

    for i in range(9):
        _, nrmse_record_src[i] = MG_SRC_Expr(
            tr_warmup_overlap=5, pred_shift=1, no_pic=True, direct_transfer=False, test_device=i
        )

        dict_nrmse_src['{}'.format(i+1)] = nrmse_record_src[i]

    # with open('F:\2021秋work\2022春work\SRC\New simulations v4\Fig3_graphs/ExprStorage_nrmse_dt.csv', mode='w', encoding='UTF-8', newline='') as f_tr:
    #     writer = csv.writer(f_tr)
    #     writer.writerows(np.array([nrmse_record_dt]))
    #
    # with open(r'F:\2021秋work\2022春work\SRC\New simulations v4\Fig3_graphs/ExprStorage_nrmse_src.csv', mode='w', encoding='UTF-8', newline='') as f_ts:
    #     writer = csv.writer(f_ts)
    #     writer.writerows(np.array([nrmse_record_src]))

    # Original device order
    plt.figure(figsize=(2.4, 1.6))
    plt.rc('font', family='Arial', size=6)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1
    device_serial = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    sns.lineplot(x=device_serial, y=nrmse_record_dt, color=np.array([247, 183, 5])/255)
    sns.lineplot(x=device_serial, y=nrmse_record_src, color=np.array([65, 176, 243])/255)
    plt.scatter(x=device_serial, s=10, y=nrmse_record_dt, color=np.array([247, 183, 5])/255, label=r'Classical framework $({\bf W}^{(9)}_{\rm out})$')
    plt.scatter(x=device_serial, s=10, y=nrmse_record_src, color=np.array([65, 176, 243])/255, label=r'TS training $({\bf W}^{(1,6,9)}_{\rm out})$')
    plt.ylim([0, 0.6])
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.legend(frameon=False, loc=2)
    plt.ylabel('NRMSE')
    plt.xlabel('Device serial')
    plt.show()

    # Rearranged I1 order
    plt.figure(figsize=(2.4, 1.6))
    plt.rc('font', family='Arial', size=6)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1

    average_record = Average_deltaI1()
    train_base = 8
    average_record = np.abs(average_record - average_record[train_base]) / average_record[train_base]

    sns.lineplot(x=average_record*100, y=nrmse_record_dt, color=np.array([247, 183, 5])/255)
    sns.lineplot(x=average_record*100, y=nrmse_record_src, color=np.array([65, 176, 243])/255)
    plt.scatter(x=average_record*100, y=nrmse_record_dt, s=10, color=np.array([247, 183, 5])/255, label=r'Classical framework $({\bf W}^{(9)}_{\rm out})$')
    plt.scatter(x=average_record*100, y=nrmse_record_src, s=10, color=np.array([65, 176, 243])/255, label=r'TS training $({\bf W}^{(1,6,9)}_{\rm out})$')
    plt.ylim([0, 0.6])
    plt.legend(frameon=False, loc=2)
    plt.ylabel('NRMSE')
    plt.xlabel('Percentage of difference in $\Delta I(t_1)$ (%)')
    plt.show()




def Average_deltaI1():
    voltages = [3]
    rounds = 50

    device_serial = ['4u', '6u', '7u', '8u', '9u', '11u', '14d', '15d', '16d']  # For 5um devices

    Average_record = np.zeros(9)
    # for i in range(devices):
    for i in range(len(device_serial)):
        for k in range(len(voltages)):
            voltage = voltages[k]
            I_0 = np.zeros(rounds)
            time_0 = np.zeros(rounds)
            I_1 = np.zeros(rounds)
            time_1 = np.zeros(rounds)
            I_2 = np.zeros(rounds)
            time_2 = np.zeros(rounds)

            df = pd.read_csv('./Pulse/' + '5um_{}_Pulse_W3R0.5_50.csv'.format(device_serial[i]),
                             header=None, sep='\n')
            df = df[0].str.split(',', expand=True)
            for j in range(rounds):
                df_0 = df.iloc[158 + 796 * j:159 + 796 * j, 1:4]
                df_0_numpy = df_0.to_numpy()

                time_0[j] = float(df_0_numpy[0, 0])
                I_0[j] = float(df_0_numpy[0, 2])

                df_1 = df.iloc[178 + 796 * j:179 + 796 * j, 1:4]
                df_1_numpy = df_1.to_numpy()
                time_1[j] = float(df_1_numpy[0, 0])
                I_1[j] = float(df_1_numpy[0, 2])

                df_2 = df.iloc[228 + 796 * j:229 + 796 * j, 1:4]
                df_2_numpy = df_2.to_numpy()
                time_2[j] = float(df_2_numpy[0, 0])
                I_2[j] = float(df_2_numpy[0, 2])

            delta_I_1 = I_0 - I_1

            Average_record[i] = np.average(delta_I_1*1e6)

    return Average_record


def Extended_Data_Figure1a():

    # Experiment
    dict_nrmse_dt = {}
    dict_nrmse_src = {}

    nrmse_record_dt = np.zeros(9)
    nrmse_record_src = np.zeros(9)

    for i in range(9):
        _, nrmse_record_dt[i] = MG_SRC_Expr(
            tr_warmup_overlap=5, pred_shift=1, no_pic=True, direct_transfer=True, test_device=i
        )

        dict_nrmse_dt['{}'.format(i+1)] = nrmse_record_dt[i]

    for i in range(9):
        _, nrmse_record_src[i] = MG_SRC_Expr(
            tr_warmup_overlap=5, pred_shift=1, no_pic=True, direct_transfer=False, test_device=i
        )

        dict_nrmse_src['{}'.format(i+1)] = nrmse_record_src[i]

    # Simulated
    levels = 9
    k3_list = np.linspace(1, 1.25, levels)
    Storage_nrmse_dt = pd.read_csv('./storage_MG_nrmse_classical.csv', header=None).values
    Storage_nrmse_src = pd.read_csv('./storage_MG_nrmse_TS.csv', header=None).values

    Diff_dict_nrmse_dt = {}

    for i in range(levels):
        Diff_dict_nrmse_dt['{}'.format(round(100 * (k3_list[i] - 1), 1))] = Storage_nrmse_dt[i, :] - Storage_nrmse_src[i, :]

    # Rearranged I1 order
    fig = plt.figure(figsize=(3.6, 2.4))
    plt.rc('font', family='Arial', size=10)
    ax = AA.Subplot(fig, 111)
    fig.add_axes(ax)

    ax.axis['left'].set_axisline_style('-|>', size=1.5)
    ax.axis['left'].line.set_color('black')
    ax.axis['left'].label.set_text(r'$\Delta$ E')
    ax.axis['top', 'right', 'bottom'].set_visible(False)
    ax.axis['x'] = ax.new_floating_axis(nth_coord=0, value=0)
    ax.axis['x'].set_axisline_style('-|>', size=1.5)
    ax.axis['x'].label.set_text('\n D2D variation strength (%)')
    ax.axis['x'].line.set_color('black')
    ax.set_xlim(0, 26)
    ax.set_ylim(-0.04, 0.1)
    ax.set_yticks(np.linspace(-0.03, 0.09, 5))
    ax.set_xticks(np.linspace(0, 25, 6))
    ax.set_xticklabels(['','5','10','15','20','25'])

    average_record = Average_deltaI1()
    train_base = 8
    average_record = np.abs(average_record - average_record[train_base]) / average_record[train_base]

    sns.lineplot(x=average_record*100, y=nrmse_record_dt**2-nrmse_record_src**2, color=np.array([65, 176, 243])/255)
    ax.scatter(x=average_record*100, y=nrmse_record_dt**2-nrmse_record_src**2, s=10, color=np.array([65, 176, 243])/255, label=r'Experiment')
    sns.lineplot(x=(k3_list-1)*100, y=np.average(Storage_nrmse_dt**2-Storage_nrmse_src**2, axis=1), color=np.array([247, 183, 5]) / 255)
    ax.scatter(x=(k3_list-1)*100, y=np.average(Storage_nrmse_dt**2-Storage_nrmse_src**2, axis=1), s=10, color=np.array([247, 183, 5]) / 255, label=r'Simulation (Average)')

    plt.legend(frameon=False, loc=2)
    plt.show()


def Extended_Data_Figure1b():
    # Only a schematic figure, the function value does not have practical significance

    x = np.linspace(0, 0.25, 26)
    y1 = 0.2/0.25*x - 0.05  # Linear approx. of the quadratic error difference
    y2 = 0.15*x**2 + y1  # The quadratic term 0.15 is far smaller than the linear term 0.8

    fig = plt.figure(figsize=(3.6, 2.4))
    plt.rc('font', family='Arial', size=10)
    ax = AA.Subplot(fig, 111)
    fig.add_axes(ax)

    ax.axis['left'].set_axisline_style('-|>', size=1.5)
    ax.axis['left'].line.set_color('black')
    ax.axis['left'].label.set_text(r'$\Delta$ E')
    ax.axis['top', 'right', 'bottom'].set_visible(False)
    ax.axis['x'] = ax.new_floating_axis(nth_coord=0, value=0)
    ax.axis['x'].set_axisline_style('-|>', size=1.5)
    ax.axis['x'].label.set_text('\n D2D variation strength')
    ax.axis['x'].line.set_color('black')
    ax.set_xlim(0, 26)
    ax.set_ylim(-0.07, 0.17)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.plot(100*x, y2, color=np.array([247, 183, 5]) / 255, linestyle='--', label='Quadratic error')
    ax.plot(100*x, y1, color=np.array([247, 183, 5]) / 255, label='Linear approximation')
    plt.legend(frameon=False, loc=2)
    plt.show()


# For Fig.3
MG_SRC_Expr(tr_warmup_overlap=5, pred_shift=1, direct_transfer=True)  # For classical framework in experiment, Fig.3c
MG_SRC_sim(direct_transfer=True)  # For classical framework in simulation, Fig.3d
MG_SRC_Expr(tr_warmup_overlap=5, pred_shift=1, direct_transfer=False)  # For TS training framework in experiment, Fig.3f
MG_SRC_sim(direct_transfer=False)  # For TS training framework in simulation, Fig.3g
NRMSE_expr()  # For NRMSE-D2D relationship plotted in Fig.3h & i
NRMSE_sim()

# NRMSE_sim() would take some time, please wait patiently (about serval hours when repeat = 50, if you wish to speed up,
# please set repeat as smaller values)
NRMSE_sim_plot()  # For Fig.3j

# For Fig.S10
NRMSE_simNoise()
NRMSE_simC2C()
# Similar to NRMSE_sim(), NRMSE_simNoise() and NRMSE_simC2C() would take some time to run, you may set repeat smaller
NRMSE_simNoisePlot()
NRMSE_simC2CPlot()

# For Fig.S11
MG_SRC_Expr_MultiChannel(direct_transfer=True)  # For multichannel RC in classical framework
MG_SRC_Expr_MultiChannel(direct_transfer=False)  # For multichannel RC in temporal switch framework

# For Extended Data Fig.1
Extended_Data_Figure1a()
Extended_Data_Figure1b()
