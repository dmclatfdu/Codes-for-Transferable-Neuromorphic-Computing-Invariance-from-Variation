# Transferable Neuromorphic Computing: Invariance from Variation
This is the readme file for the codes for Transferable Neuromorphic Computing: Invariance from Variation.

There are six code files and a data file in total, which are:

base_library.py
sim_RC_library.py
device_characteristics.py
RC_MG.py
RC_Lorenz.py
RC_arrhythmia.py
ECGdataset.mat

Before you run any of these files please make sure that they are all in the directory '/Source data', the data folder we shared.
Eg., '/Source data/RC_MG.py'

The base_library.py stores the basic functions used in all other programs.

The sim_RC_library stores the initial settings of the simulated TiOx-based RC and the related pre-processing procedures.

The device_character.py stores the analysis of experiment data and the corresponding reproduction through simulation.

The RC_MG.py stores all programs related to the MG one-step prediction in the paper.

The RC_Lorenz.py stores the programs for the Lorenz recurrent prediction.

The RC_arrhythmia.py stores the programs for arrhythmia detection task based on the MITBIH database. The code is modified from that from Codes in NE2022 of https://github.com/Tsinghua-LEMON-Lab/Reservoir-computing

#NOTE: The code in RC_arrhythmia.py would take a lot of time, about 3-4 days on a desktop CPU (we use the AMD Ryzen 5800H).#
#NOTE: PLEASE RUN THE CODES AS THE WAY THEY ARE ORDERED TO ENSURE THE FILES NEEDED FOR THE FOLLOWING STEPS ARE CREATED#

The ECGdataset.mat is the ECG dataset of heartbeat records, with reference to MITBIH Arrhythmia Database and research paper, A memristor-based analogue reservoir computing system for real-time and power-efficient signal processing, published in Nature Electronics.

To run the codes, the following libraries are required:
SciPy (1.7.1), tqdm (4.62.3), NumPy (1.22.4), Pandas (1.3.4), Seaborn (0.12.2), Matplotlib (3.4.3), Scikit-Learn (1.0.1), h5py (3.7.0)

