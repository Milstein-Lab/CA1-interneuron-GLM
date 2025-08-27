import sys
import os

sys.path.append(r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM")
from GLM_regression import *
from functions_plotting_reconstruction_MSE import *
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import slicetca
import h5py
import utils as ut
import plot as pt
import warnings
plt.rcParams.update({'font.size': 12,
                     'axes.spines.right': False,
                     'axes.spines.top':   False,
                     'legend.frameon':    False,})

# %load_ext autoreload
# %autoreload


def compare_belts(activity_dict_NDNF_newest):
    belt_E_dict = {}
    belt_1A_dict = {}
    belt_1B_dict = {}

    for idx, animal in enumerate(activity_dict_NDNF_newest):
        if idx < 18:
            cell_dict = {}
            for cell in activity_dict_NDNF_newest[animal]:
                cell_dict[cell] = activity_dict_NDNF_newest[animal][cell]
            belt_E_dict[animal] = cell_dict
        elif 17 < idx < 31:
            cell_dict = {}
            for cell in activity_dict_NDNF_newest[animal]:
                cell_dict[cell] = activity_dict_NDNF_newest[animal][cell]
            belt_1A_dict[animal] = cell_dict
        elif idx > 30:
            cell_dict = {}
            for cell in activity_dict_NDNF_newest[animal]:
                cell_dict[cell] = activity_dict_NDNF_newest[animal][cell]
            belt_1B_dict[animal] = cell_dict

    animal_list = []

    for animal in belt_1B_dict:
        for cell in belt_1B_dict[animal]:
            animal_list.append(np.mean(belt_1B_dict[animal][cell], axis=1))

    animal_array = np.array(animal_list)

    mean_animal_array_1B = np.mean(animal_array, axis=0)
    sem_animal_array_1B = sem(animal_array, axis=0)

    plt.plot(mean_animal_array_1B, label='Track 1B')
    plt.fill_between(range(len(mean_animal_array_1B)), mean_animal_array_1B + sem_animal_array_1B, mean_animal_array_1B - sem_animal_array_1B, alpha=0.2)

    animal_list = []

    for animal in belt_1A_dict:
        cell_list = []
        for cell in belt_1A_dict[animal]:
            animal_list.append(np.mean(belt_1A_dict[animal][cell], axis=1))

    animal_array = np.array(animal_list)

    mean_animal_array_1A = np.mean(animal_array, axis=0)
    sem_animal_array_1A = sem(animal_array, axis=0)

    plt.plot(mean_animal_array_1A, label="Track 1A")
    plt.fill_between(range(len(mean_animal_array_1A)), mean_animal_array_1A + sem_animal_array_1A, mean_animal_array_1A - sem_animal_array_1A, alpha=0.2)
    plt.title("Cell Average Cue vs Fixed Reward Raw")
    plt.legend()
    plt.show()


def plot_NDNF_velocity(filtered_factors_dict_NDNF_newest):

    belt_E_dict_vel = {}
    belt_1A_dict_vel = {}
    belt_1B_dict_vel = {}

    for idx, animal in enumerate(filtered_factors_dict_NDNF_newest):
        if idx < 18:
            belt_E_dict_vel[animal] = filtered_factors_dict_NDNF_newest[animal]['Velocity']
        elif 17 < idx < 31:
            belt_1A_dict_vel[animal] = filtered_factors_dict_NDNF_newest[animal]['Velocity']
        elif idx > 30:
            belt_1B_dict_vel[animal] = filtered_factors_dict_NDNF_newest[animal]['Velocity']


    mean_list_1A = []
    for animal in belt_1A_dict_vel:
        mean_list_1A.append(np.mean(belt_1A_dict_vel[animal], axis=1))

    list_1A_array = np.array(mean_list_1A)


    mean_list_1A_array = np.mean(list_1A_array, axis=0)
    sem_list_1A_array = sem(list_1A_array, axis=0)


    mean_list_1B = []
    for animal in belt_1B_dict_vel:
        mean_list_1B.append(np.mean(belt_1B_dict_vel[animal], axis=1))

    list_1B_array = np.array(mean_list_1B)

    mean_list_1B_array = np.mean(list_1B_array, axis=0)
    sem_list_1B_array = sem(list_1B_array, axis=0)

    plt.plot(mean_list_1A_array, color='r', label='Fixed Reward')
    plt.fill_between(range(len(mean_list_1A_array)), mean_list_1A_array+sem_list_1A_array, mean_list_1A_array-sem_list_1A_array, alpha=0.2, color='r')
    plt.plot(mean_list_1B_array, color='b', label='Fixed Reward + Cue')
    plt.fill_between(range(len(mean_list_1B_array)), mean_list_1B_array+sem_list_1B_array, mean_list_1B_array-sem_list_1B_array, alpha=0.2, color='b')
    plt.legend()
    plt.xlabel("Position Bins")
    plt.ylabel("meters/sec")
    plt.title("Aniaml Velocity")
    plt.show()