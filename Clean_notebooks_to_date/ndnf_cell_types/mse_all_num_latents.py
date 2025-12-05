import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy.stats import sem
import glob


def get_data_by_animal_num(base_dir, file_name_template, animal_list=None, num_animals=None):
    SST_list_2_22 = []

    if animal_list is None:
        animal_list = range(num_animals)

    for i in animal_list:
        file_name = file_name_template.format(i=i)
        file_path = os.path.join(base_dir, file_name)

        with open(file_path, "rb") as f:
            animal = pickle.load(f)

        SST_list_2_22.append(animal)

    # SST_mean_2_22, SST_sem_2_22 = get_mean_sem_loss2(SST_list_2_22)

    SST_list_2_22_array = np.array(SST_list_2_22)

    SST_mean_2_22 = np.mean(SST_list_2_22_array, axis=0)
    SST_sem_2_22 = sem(SST_list_2_22_array, axis=0)



    return SST_mean_2_22, SST_sem_2_22


def cell_type_get_mean_sem(base_dir, start_num=None, end_num=None, cell_type=None):
    latent_numbers = []

    for i in range(start_num, end_num):
        pattern = os.path.join(base_dir, f"loss_dict_EC_latent_{i}_{i}_animal*.pkl")
        file_paths = sorted(glob.glob(pattern))

        loss_floats = []

        for path in file_paths:
            with open(path, "rb") as f:
                data = pickle.load(f)
                data = float(data)
                loss_floats.append(data)
        latent_numbers.append(loss_floats)

    EC_latent_means_42_62 = []
    EC_latent_sems_42_62 = []

    for i in latent_numbers:
        EC_latent_means_42_62.append(np.mean(i))
        EC_latent_sems_42_62.append(sem(i))

    return EC_latent_means_42_62, EC_latent_sems_42_62


def load_data_MSEs():
    # base_dir = "Users/Msfin/cloned_repositories/CA1-interneuron-GLM/datasets"

    base_dir = "/Users/michaelfinch/CA1-interneuron-GLM/datasets/SliceTCA_MSE_Reconstruction/"

    file_name_template = "loss_dict_SST_latent_2_22_animal{i}"
    SST_mean_2_22, SST_sem_2_22 = get_data_by_animal_num(base_dir, file_name_template, animal_list=None, num_animals=10)

    # base_dir = r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets"
    file_name_template = "loss_dict_SST_latent_22_42_animal{i}"
    SST_mean_22_42, SST_sem_22_42 = get_data_by_animal_num(base_dir, file_name_template, animal_list=None, num_animals=10)

    # base_dir = r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets"
    file_name_template = "loss_dict_SST_latent_42_62_animal{i}"
    sst_mean_42_64, sst_sem_42_64 = get_data_by_animal_num(base_dir, file_name_template, animal_list=None, num_animals=10)


    # base_dir = r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets"
    file_name_template = "NDNF_latent_loss_2_22_animal_{i}"
    NDNF_mean_2_22, NDNF_sem_2_22 = get_data_by_animal_num(base_dir, file_name_template, animal_list=None, num_animals=4)

    # base_dir = r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets"
    file_name_template = "loss_dict_NDNF_latent_22_42_animal{i}"
    NDNF_mean_22_42, NDNF_sem_22_42 = get_data_by_animal_num(base_dir, file_name_template, animal_list=None, num_animals=4)

    # base_dir = r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets"
    file_name_template = "loss_dict_NDNF_latent_42_62_animal{i}"
    animal_list = [0, 2, 3]
    NDNF_mean_42_62, NDNF_sem_42_62 = get_data_by_animal_num(base_dir, file_name_template, animal_list=animal_list, num_animals=None)



    base_dir = "/Users/michaelfinch/CA1-interneuron-GLM/datasets/SliceTCA_MSE_Reconstruction/EC_experiment"

    EC_latent_means_2_22, EC_latent_sems_2_22 = cell_type_get_mean_sem(base_dir, start_num=2, end_num=22, cell_type="EC")

    # base_dir = r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets"
    base_dir = "/Users/michaelfinch/CA1-interneuron-GLM/datasets/SliceTCA_MSE_Reconstruction"
    EC_latent_means_22_42, EC_latent_sems_22_42 = cell_type_get_mean_sem(base_dir, start_num=22, end_num=43, cell_type="EC")

    base_dir = "/Users/michaelfinch/CA1-interneuron-GLM/datasets/SliceTCA_MSE_Reconstruction/EC_experiment"
    # base_dir = r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets\EC_experiment"
    EC_latent_means_42_62, EC_latent_sems_42_62 = cell_type_get_mean_sem(base_dir, start_num=43, end_num=63, cell_type="EC")



    print(f"EC_latent_means_2_22 {EC_latent_means_2_22}")



    concatenated_EC_means = np.concatenate([EC_latent_means_2_22, EC_latent_means_22_42, EC_latent_means_42_62])
    concatenated_EC_sem = np.concatenate([EC_latent_sems_2_22, EC_latent_sems_22_42, EC_latent_sems_42_62])

    concatenated_SST_means = np.concatenate([SST_mean_2_22, SST_mean_22_42[:-1], sst_mean_42_64[:-1]])
    concatenated_SST_sem = np.concatenate([SST_sem_2_22, SST_sem_22_42[:-1], sst_sem_42_64[:-1]])

    concatenated_NDNF_means = np.concatenate([NDNF_mean_2_22, NDNF_mean_22_42[:-1], NDNF_mean_42_62[:-1]])
    concatenated_NDNF_sem = np.concatenate([NDNF_sem_2_22, NDNF_sem_22_42[:-1], NDNF_sem_42_62[:-1]])



    ##########################################




    # base_dir = r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets\EC_timebins_experiment"

    # # Holds lists of scalar losses for each latent number
    # latent_numbers = []

    # for i in range(2, 63):
    #     # Get all matching files
    #     pattern = os.path.join(base_dir, f"timebins_loss_dict_EC_latent_{i}_{i}_animal*.pkl")
    #     file_paths = sorted(glob.glob(pattern))

    #     # Store float values from each file
    #     loss_floats = []

    #     for path in file_paths:
    #         with open(path, "rb") as f:
    #             data = pickle.load(f)
    #             data = float(data)
    #             loss_floats.append(data)

    #     latent_numbers.append(loss_floats)

    # EC_latent_means_timebins_62 = []
    # EC_latent_sems_timebins_62 = []

    # for i in latent_numbers:
    #     EC_latent_means_timebins_62.append(np.mean(i))
    #     EC_latent_sems_timebins_62.append(sem(i))

    # EC_latent_sems_timebins_62 = np.array(EC_latent_sems_timebins_62)








    # base_dir = r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets\EC_timebins_experiment"

    # # Holds lists of scalar losses for each latent number
    # latent_numbers = []

    # for i in range(2, 63):
    #     # Get all matching files
    #     pattern = os.path.join(base_dir, f"timebins_loss_dict_NDNF_latent_{i}_{i}_animal*.pkl")
    #     file_paths = sorted(glob.glob(pattern))

    #     # Store float values from each file
    #     loss_floats = []

    #     for path in file_paths:
    #         with open(path, "rb") as f:
    #             data = pickle.load(f)
    #             data = float(data)
    #             loss_floats.append(data)

    #     latent_numbers.append(loss_floats)

    # NDNF_latent_means_timebins_62 = []
    # NDNF_latent_sems_timebins_62 = []

    # for i in latent_numbers:
    #     NDNF_latent_means_timebins_62.append(np.mean(i))
    #     NDNF_latent_sems_timebins_62.append(sem(i))
    # NDNF_latent_sems_timebins_62 = np.array(NDNF_latent_sems_timebins_62)



    # # base_dir = r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets"
    # base_dir = "/Users/michaelfinch/CA1-interneuron-GLM/datasets/SliceTCA_MSE_Reconstruction/"

    # SST_list_2_24_timebins = []


    # for i in range(10):  # Assuming animals 0 to

    #     file_name = f"timebins_loss_dict_SST_latent_2_22_animal{i}.pkl"
    #     file_path = os.path.join(base_dir, file_name)

    #     # Load the pickle file
    #     with open(file_path, "rb") as f:
    #         animal = pickle.load(f)

    #     SST_list_2_24_timebins.append(animal)

    
    # SST_list_2_24_timebins_array = np.array(SST_list_2_24_timebins)

    # SST_mean_2_24_timebins = np.mean(SST_list_2_24_timebins_array, axis=0)
    # SST_sem_2_24_timebins = sem(SST_list_2_24_timebins_array, axis=0)

    # # SST_mean_2_24_timebins, SST_sem_2_24_timebins = get_mean_sem_loss2(SST_list_2_24_timebins)


    # base_dir = r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets"

    # SST_list_22_42_timebins = []


    # for i in range(10):  # Assuming animals 0 to

    #     file_name = f"timebins_loss_dict_SST_latent_22_42_animal{i}.pkl"
    #     file_path = os.path.join(base_dir, file_name)

    #     # Load the pickle file
    #     with open(file_path, "rb") as f:
    #         animal = pickle.load(f)

    #     SST_list_22_42_timebins.append(animal)


    # SST_list_22_42_timebins_array = np.array(SST_list_22_42_timebins)

    # SST_mean_22_42_timebins = np.mean(SST_list_22_42_timebins_array, axis=0)
    # SST_sem_22_42_timebins = sem(SST_list_22_42_timebins_array, axis=0)


    # # SST_mean_22_42_timebins, SST_sem_22_42_timebins = get_mean_sem_loss2(SST_list_22_42_timebins)


    # base_dir = r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets"

    # SST_list_42_62_timebins = []


    # for i in range(10):  # Assuming animals 0 to

    #     file_name = f"timebins_loss_dict_SST_latent_42_62_animal{i}.pkl"
    #     file_path = os.path.join(base_dir, file_name)

    #     # Load the pickle file
    #     with open(file_path, "rb") as f:
    #         animal = pickle.load(f)

    #     SST_list_42_62_timebins.append(animal)

    # SST_list_42_62_timebins_array = np.array(SST_list_42_62_timebins)

    # SST_mean_42_62_timebins = np.mean(SST_list_42_62_timebins_array, axis=0)
    # SST_sem_42_62_timebins = sem(SST_list_42_62_timebins_array, axis=0)

    # # SST_mean_42_62_timebins, SST_sem_42_62_timebins = get_mean_sem_loss2(SST_list_42_62_timebins)

    # SST_latent_mean_timebins_62 = np.concatenate([SST_mean_2_24_timebins, SST_mean_22_42_timebins, SST_mean_42_62_timebins])
    # SST_latent_sem_timebins_62 = np.concatenate([SST_sem_2_24_timebins, SST_sem_22_42_timebins, SST_sem_42_62_timebins])


    return concatenated_EC_means, concatenated_EC_sem, concatenated_SST_means, concatenated_SST_sem, concatenated_NDNF_means, concatenated_NDNF_sem #SST_latent_mean_timebins_62, SST_latent_sem_timebins_62, NDNF_latent_means_timebins_62, NDNF_latent_sems_timebins_62, EC_latent_means_timebins_62, EC_latent_sems_timebins_62


def plot_MSE_latents(concatenated_EC_means, concatenated_EC_sem, concatenated_SST_means, concatenated_SST_sem, concatenated_NDNF_means, concatenated_NDNF_sem, SST_latent_mean_timebins_62, SST_latent_sem_timebins_62, NDNF_latent_means_timebins_62, NDNF_latent_sems_timebins_62, EC_latent_means_timebins_62, EC_latent_sems_timebins_62):

    ##########################################

    fig, axs = plt.subplots(1,2, figsize=(20,10))

    axs[0].plot(concatenated_EC_means, label="EC", color='g')
    axs[0].fill_between(range(len(concatenated_EC_means)),
                     concatenated_EC_means + concatenated_EC_sem,
                     concatenated_EC_means - concatenated_EC_sem,
                     alpha=0.3, color='g')
    axs[0].plot(concatenated_SST_means, label="SST", color='b')
    axs[0].fill_between(range(len(concatenated_SST_means)),
                     concatenated_SST_means + concatenated_SST_sem,
                     concatenated_SST_means - concatenated_SST_sem,
                     alpha=0.3, color='b')

    axs[0].plot(concatenated_NDNF_means, label="NDNF", color='orange')
    axs[0].fill_between(range(len(concatenated_NDNF_means)),
                     concatenated_NDNF_means + concatenated_NDNF_sem,
                     concatenated_NDNF_means - concatenated_NDNF_sem,
                     alpha=0.3, color='orange')
    axs[0].legend()
    axs[0].set_ylabel("MSE")
    axs[0].set_ylim(0,1)
    axs[0].set_ylabel("MSE")
    axs[0].set_title("Slice TCA per Animal (x,0,0)")
    axs[0].set_xlabel("Number of Latents")


    axs[1].plot(SST_latent_mean_timebins_62, color='b', label='SST')
    axs[1].fill_between(range(len(SST_latent_mean_timebins_62)),
                     SST_latent_mean_timebins_62 + SST_latent_sem_timebins_62,
                     SST_latent_mean_timebins_62 - SST_latent_sem_timebins_62,
                     alpha=0.3, color='b')
    axs[1].plot(NDNF_latent_means_timebins_62, color='orange', label='NDNF')
    axs[1].fill_between(range(len(NDNF_latent_means_timebins_62)),
                     NDNF_latent_means_timebins_62 + NDNF_latent_sems_timebins_62,
                     NDNF_latent_means_timebins_62 - NDNF_latent_sems_timebins_62,
                     alpha=0.3, color='orange')
    axs[1].plot(EC_latent_means_timebins_62, color='g', label='EC')
    axs[1].fill_between(range(len(EC_latent_means_timebins_62)),
                     EC_latent_means_timebins_62 + EC_latent_sems_timebins_62,
                     EC_latent_means_timebins_62 - EC_latent_sems_timebins_62,
                     alpha=0.3, color='g')
    axs[1].legend()
    axs[1].set_ylabel("MSE")
    axs[1].set_title("Slice TCA per Animal (0,0,x)")
    axs[1].set_ylim(0,1)
    axs[1].set_xlabel("Number of Latents")

    plt.show()


def run():
    concatenated_EC_means, concatenated_EC_sem, concatenated_SST_means, concatenated_SST_sem, concatenated_NDNF_means, concatenated_NDNF_sem = load_data_MSEs() #SST_latent_mean_timebins_62, SST_latent_sem_timebins_62, NDNF_latent_means_timebins_62, NDNF_latent_sems_timebins_62, EC_latent_means_timebins_62, EC_latent_sems_timebins_62 = load_data_MSEs()

    # plot_MSE_latents(concatenated_EC_means, concatenated_EC_sem, concatenated_SST_means, concatenated_SST_sem, concatenated_NDNF_means, concatenated_NDNF_sem, SST_latent_mean_timebins_62, SST_latent_sem_timebins_62, NDNF_latent_means_timebins_62, NDNF_latent_sems_timebins_62, EC_latent_means_timebins_62, EC_latent_sems_timebins_62)

    
    print(f"concatenated_EC_means {concatenated_EC_means}")

    plt.plot(concatenated_EC_means, label="EC", color='g')
    plt.fill_between(range(len(concatenated_EC_means)),
                     concatenated_EC_means + concatenated_EC_sem,
                     concatenated_EC_means - concatenated_EC_sem,
                     alpha=0.3, color='g')
    
    concatenated_SST_mean = np.mean(concatenated_SST_means, axis=(1,2,3))
    concatenated_SST_sems = np.mean(concatenated_SST_sem, axis=(1,2,3))
    plt.plot(concatenated_SST_mean, label="SST", color='b')
    plt.fill_between(range(len(concatenated_SST_mean)),
                     concatenated_SST_mean + concatenated_SST_sems,
                     concatenated_SST_mean - concatenated_SST_sems,
                     alpha=0.3, color='b')

    concatenated_NDNF_mean = np.mean(concatenated_NDNF_means, axis=(1,2,3))
    concatenated_NDNF_sems = np.mean(concatenated_SST_sem, axis=(1,2,3))
    plt.plot(concatenated_NDNF_mean, label="NDNF", color='orange')
    plt.fill_between(range(len(concatenated_NDNF_mean)),
                     concatenated_NDNF_mean + concatenated_NDNF_sems,
                     concatenated_NDNF_mean - concatenated_NDNF_sems,
                     alpha=0.3, color='orange')
    plt.legend()
    plt.ylabel("MSE")
    plt.ylim(0,1)
    plt.ylabel("MSE")
    plt.title("Slice TCA per Animal (x,0,0)")
    plt.xlabel("Number of Latents")

    plt.tight_layout()
    plt.show()

if __name__ =="__main__":
    run()
    