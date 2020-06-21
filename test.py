import sys
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.nn.functional as F

use_cuda = False
print(use_cuda)

import numpy as np
from tqdm import tqdm, trange
import pickle
from rdkit import Chem, DataStructs
from release.stackRNN import StackAugmentedRNN
from release.data import GeneratorData
from release.utils import canonical_smiles


import matplotlib.pyplot as plt
import seaborn as sns

print("finished imports")

gen_data_path = './data/chembl_22_clean_1576904_sorted_std_final.smi'
test_gen_data_path = './data/testingdata.smi'

tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
          '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
          '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']

gen_data = GeneratorData(training_data_path=test_gen_data_path, delimiter='\t',
                         cols_to_read=[0], keep_header=True, tokens=tokens)



def plot_hist(prediction, n_to_generate):
    print("Mean value of predictions:", prediction.mean())
    print("Proportion of valid SMILES:", len(prediction)/n_to_generate)
    ax = sns.kdeplot(prediction, shade=True)
    ax.set(xlabel='Predicted pIC50',
           title='Distribution of predicted pIC50 for generated molecules')
    plt.show()



def estimate_and_update(generator, predictor, n_to_generate, **kwargs):
    generated = []
    pbar = tqdm(range(n_to_generate))
    for i in pbar:
        pbar.set_description("Generating molecules...")
        generated.append(generator.evaluate(gen_data, predict_len=120)[1:-1])

    sanitized = canonical_smiles(generated, sanitize=False, throw_warning=False)[:-1]
    unique_smiles = list(np.unique(sanitized))[1:]
    smiles, prediction, nan_smiles = predictor.predict(unique_smiles, get_features=get_fp)

    plot_hist(prediction, n_to_generate)

    return smiles, prediction





hidden_size = 1500
stack_width = 1500
stack_depth = 200
layer_type = 'GRU'
lr = 0.001
optimizer_instance = torch.optim.Adadelta

my_generator = StackAugmentedRNN(input_size=gen_data.n_characters, hidden_size=hidden_size,
                                 output_size=gen_data.n_characters, layer_type=layer_type,
                                 n_layers=1, is_bidirectional=False, has_stack=True,
                                 stack_width=stack_width, stack_depth=stack_depth,
                                 use_cuda=use_cuda,
                                 optimizer_instance=optimizer_instance, lr=lr)
print("StackRNN initialized")

model_path = './checkpoints/generator/checkpoint_biggest_rnn'

my_generator.load_model(model_path)

print("training finished")


print(my_generator.evaluate())



from release.utils import get_desc, get_fp
from mordred import Calculator, descriptors


# calc = Calculator(descriptors, ignore_3D=True)

# pred_data = PredictorData(path='./data/jak2_data.csv', get_features=get_fp)

# print("Predictor initialzied")
# from release.predictor import VanillaQSAR
# from sklearn.ensemble import RandomForestRegressor as RFR

# model_instance = RFR
# model_params = {'n_estimators': 250, 'n_jobs': 10}

# my_predictor = VanillaQSAR(model_instance=model_instance,
#                            model_params=model_params,
#                            model_type='regressor')

# my_predictor.fit_model(pred_data, cv_split='random')

# print("predictor fitted")


# from release.reinforcement import Reinforcement
# my_generator_max = StackAugmentedRNN(input_size=gen_data.n_characters,
#                                      hidden_size=hidden_size,
#                                      output_size=gen_data.n_characters,
#                                      layer_type=layer_type,
#                                      n_layers=1, is_bidirectional=False, has_stack=True,
#                                      stack_width=stack_width, stack_depth=stack_depth,
#                                      use_cuda=use_cuda,
#                                      optimizer_instance=optimizer_instance, lr=lr)
# my_generator_max.load_model(model_path)
# print("Max generator initialzied")
#
#
# #Experiment parameters
# n_to_generate = 200
# n_policy_replay = 10
# n_policy = 15
# n_iterations = 100
#
#
# def simple_moving_average(previous_values, new_value, ma_window_size=10):
#     value_ma = np.sum(previous_values[-(ma_window_size-1):]) + new_value
#     value_ma = value_ma/(len(previous_values[-(ma_window_size-1):]) + 1)
#     return value_ma
#
# def get_reward_max(smiles, predictor, invalid_reward=0.0, get_features=get_fp):
#     mol, prop, nan_smiles = predictor.predict([smiles], get_features=get_features)
#     if len(nan_smiles) == 1:
#         return invalid_reward
#     return np.exp(prop[0]/3)
#
# x = np.linspace(0, 12)
# y = np.exp(x/3)
# plt.plot(x, y)
# plt.xlabel('pIC50 value')
# plt.ylabel('Reward value')
# plt.title('Reward function for JAK2 activity maximization')
# plt.show()
