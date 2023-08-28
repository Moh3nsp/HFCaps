from audioop import avg

import torch

import torch.nn as nn
from torch.utils.data import DataLoader
from torch import tensor
from fedcaps import LocalFedCapsNet, fed_train, test
from database import get_train_dataset, get_testset_dataloader
from heterogen_fedcaps import GlobalFedCaps, LocalFedCapsNet,  test ,fed_train
import math
from typing import List
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import copy
import torch.nn.functional as F

from torch.autograd import Variable


def clustering(models: List[torch.nn.Module], digit_caps_name='digitCaps.routing_module.b', class_no=10):
    model_dicts = {i: [] for i in range(class_no)}
    model_array = []
    vector_dicts = {i: [] for i in range(len(models))}
    # concate each models DigitCaps
    # return a dic of clustered models
    pca = PCA(n_components=10)
    for index in range(len(models)):
        flattened = np.array(models[index].state_dict()[digit_caps_name].flatten().cpu())
        vector_dicts[index] = flattened
        model_array.append(flattened)
    kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto").fit(model_array)
    clusters_index = kmeans.predict(model_array)
    clusters = {i: [] for i in set(clusters_index)}
    for index in range(len(clusters_index)):
        clusters[clusters_index[index]].append(index)
    return clusters

def get_conv1_padded(original_model, giant_model):
    weights = original_model.state_dict()['conv1.weight']
    gaint_shape = giant_model.state_dict()['conv1.weight'].shape
    shape= weights.shape
    remained_size = gaint_shape[0] - shape[0]
    padded_weights = F.pad(input=weights, pad=(0, 0, 0, 0, 0, 0, 0, remained_size), value=0)
    return padded_weights

def get_primary_caps_padded(original_model, giant_model):
    weights = original_model.state_dict()['primaryCaps.conv.weight']
    gaint_shape = giant_model.state_dict()['primaryCaps.conv.weight'].shape
    shape= weights.shape
    remained_size_d0 = gaint_shape[0] - shape[0]
    remained_size_d1 = gaint_shape[1] - shape[1]
    padded_weights = F.pad(input=weights, pad=(0,0,0,0,0,remained_size_d1,0,remained_size_d0), value=0)
    return padded_weights


def get_primary_caps_bias_padded(original_model, giant_model):
    weights = original_model.state_dict()['primaryCaps.conv.bias']
    gaint_shape = giant_model.state_dict()['primaryCaps.conv.bias'].shape
    shape= weights.shape
    remained_size_d0 = gaint_shape[0] - shape[0]
    padded_weights = F.pad(input=weights, pad=(0,remained_size_d0), value=0)
    return padded_weights



def create_globalmodel(local_models, global_model: GlobalFedCaps):
    # Create Global Model (Final Product)
    # Replace Models weights
    # odict_keys(['conv_1.0.weight', 'conv_1.0.bias', 'conv_1.1.weight', 'conv_1.1.bias', 'conv_1.2.weight', 'conv_1.2.bias', 'conv_1.3.weight', 'conv_1.3.bias', 'conv_1.4.weight', 'conv_1.4.bias', 'conv_1.5.weight', 'conv_1.5.bias', 'conv_1.6.weight', 'conv_1.6.bias', 'conv_1.7.weight', 'conv_1.7.bias', 'conv_1.8.weight', 'conv_1.8.bias', 'conv_1.9.weight', 'conv_1.9.bias', 'conv_1.10.weight', 'conv_1.10.bias', 'conv_1.11.weight', 'conv_1.11.bias', 'conv_1.12.weight', 'conv_1.12.bias', 'conv_1.13.weight', 'conv_1.13.bias', 'conv_1.14.weight', 'conv_1.14.bias', 'conv_1.15.weight', 'conv_1.15.bias', 'conv_1.16.weight', 'conv_1.16.bias', 'conv_1.17.weight', 'conv_1.17.bias', 'conv_1.18.weight', 'conv_1.18.bias', 'conv_1.19.weight', 'conv_1.19.bias', 'primaryCaps.0.conv.weight', 'primaryCaps.0.conv.bias', 'primaryCaps.1.conv.weight', 'primaryCaps.1.conv.bias',
    # 'primaryCaps.2.conv.weight', 'primaryCaps.2.conv.bias', 'primaryCaps.3.conv.weight', 'primaryCaps.3.conv.bias', 'primaryCaps.4.conv.weight', 'primaryCaps.4.conv.bias', 'primaryCaps.5.conv.weight', 'primaryCaps.5.conv.bias', 'primaryCaps.6.conv.weight',
    # 'primaryCaps.6.conv.bias', 'primaryCaps.7.conv.weight', 'primaryCaps.7.conv.bias', 'primaryCaps.8.conv.weight', 'primaryCaps.8.conv.bias', 'primaryCaps.9.conv.weight', 'primaryCaps.9.conv.bias', 'primaryCaps.10.conv.weight', 'primaryCaps.10.conv.bias', 'primaryCaps.11.conv.weight', 'primaryCaps.11.conv.bias', 'primaryCaps.12.conv.weight', 'primaryCaps.12.conv.bias', 'primaryCaps.13.conv.weight', 'primaryCaps.13.conv.bias', 'primaryCaps.14.conv.weight', 'primaryCaps.14.conv.bias', 'primaryCaps.15.conv.weight', 'primaryCaps.15.conv.bias', 'primaryCaps.16.conv.weight', 'primaryCaps.16.conv.bias', 'primaryCaps.17.conv.weight', 'primaryCaps.17.conv.bias', 'primaryCaps.18.conv.weight', 'primaryCaps.18.conv.bias', 'primaryCaps.19.conv.weight', 'primaryCaps.19.conv.bias', 'routing_module.b', 'digitCaps.weights', 'digitCaps.routing_module.b'])
    orginal_ensembleCount = len(local_models)
    for index in range(len(local_models)):
        global_model.state_dict()[f'conv_1.{index}.weight'].copy_(
            local_models[index].state_dict()['conv1.weight'].clone().detach())

        global_model.state_dict()[f'conv_1.{index}.bias'].copy_(
            local_models[index].state_dict()['conv1.bias'].clone().detach())

        global_model.state_dict()[f'primaryCaps.{index}.conv.weight'].copy_(
            local_models[index].state_dict()['primaryCaps.conv.weight'].clone().detach())

        global_model.state_dict()[f'primaryCaps.{index}.conv.bias'].copy_(
            local_models[index].state_dict()['primaryCaps.conv.bias'].clone().detach())

    list_digitcaps_weights = [model.state_dict(
    )['digitCaps.weights'].clone().detach() for model in local_models]
    concated_digitcaps_weights = torch.cat(tuple(list_digitcaps_weights), 0)

    list_digitcaps_bias_weights = [model.state_dict(
    )['digitCaps.routing_module.b'].clone().detach() for model in local_models]
    concated_digitcaps_bias_weights = torch.cat(
        tuple(list_digitcaps_bias_weights), 0)

    routing_module_bias_weights = [model.state_dict(
    )['routing_module.b'].clone().detach() for model in local_models]
    concated_routing_module_weights = torch.cat(
        tuple(routing_module_bias_weights), 0)

    # apply concatenated capsules to global model
    global_model.state_dict()['routing_module.b'].copy_(
        concated_routing_module_weights)
    global_model.state_dict()['digitCaps.weights'].copy_(
        concated_digitcaps_weights)
    global_model.state_dict()['digitCaps.routing_module.b'].copy_(
        concated_digitcaps_bias_weights)
    return global_model


def clustering_model_by_digitcaps(model, str_digit_caps_bias_key, classNo):
    weights = model.state_dict()[str_digit_caps_bias_key].puermte(2, 0, 1)
    norms = []
    for i in range(classNo):
        norms.append(torch.norm(weights[i * 16:(i + 1) * 16].flatten()))
    cluster = torch.argmax(torch.tensor(norms))
    return cluster, norms[cluster]

def clustering_model_by_digitcapsv2(model, str_digit_caps_bias_key, classNo):
    weights = model.state_dict()[str_digit_caps_bias_key].permute(1,0)
    norms = []
    for i in range(classNo):
        norms.append(torch.norm(weights[i] ))
    cluster = torch.argmax(torch.tensor(norms))
    return cluster, norms[cluster]

def fed_avg(models_arr, is_fedCaps: bool = False):
    if is_fedCaps == False:
        main = models_arr[0].state_dict()
    else:
        main = models_arr[0]
    keys = main.keys()
    for index in range(len(models_arr)):
        if (index > 0):
            current_model = models_arr[index]
            for key in keys:
                if is_fedCaps == False:
                    main[key] += current_model.state_dict()[key]
                else:
                    main[key] += current_model[key]
    for key in keys:
        main[key] /= len(models_arr)
    return main



################# Fixed #######################


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
non_iid_percentage = 90
routing_iterations = 3
class_number = 10
avg_loss_margin = 0.1
sample_num = 50
attackers_per_communication = []
is_fashion_dataset = True
if (is_fashion_dataset):
    from database import FashionMNIST_get_train_dataset as get_train_dataset, \
        FashionMNIST_get_testset_dataloader as get_testset_dataloader

##################################################

participant_number = tensor(10)

reconstruction_alpha = 0.0001
landa = 0.5
global_acc = []
avg_acc = []

model_configs =[ (10, 3, 8, 9, 9)
                ,(20, 10, 8, 9, 9)
                ,(30, 17, 8, 9, 9)
                ,(40, 24, 8, 9, 9)
                ,(50, 31, 8, 9, 9)
                ,(60, 38, 8, 9, 9)
                ,(70, 45, 8, 9, 9)
                ,(80, 52, 8, 9, 9)
                ,(90, 69, 8, 9, 9)
                ,(100, 76, 8, 9, 9)
                 ]



giantCaps_local_models = [ LocalFedCapsNet(*conf).to(device)  for conf in model_configs]


# dataset load
datasets = []

random_percentage = 80
random_clas_number = int(((random_percentage / 100) * class_number))

bias_labels = []

random_labels = []

total_random_count = 0

final_labels = [0,1,2,3,4,5,6,7,8,9]

# final_labels = torch.randint(0, 10, size=(100,)).data

for lbl_name in final_labels:
    datasets.append(get_train_dataset(
        lbl_name, sample_num, non_iid_percentage))

datanoise_attacks_count = 0
datanoise_attacks_Index = []

# load training test set

clustered_models = {i: [] for i in range(class_number)}
clustered_models_single = {i: [] for i in range(class_number)}

total_comunications = 100
is_first_comunicate = True
local_training_count = 1
is_first_comunicate_Fedcaps_Agg = True
global_total_acc = []
fed_caps_final_acc_ensemble = []
clustered_model = {_: [] for _ in range(class_number)}

global_model_test = get_testset_dataloader(800)

global_test_set_dataloader = DataLoader(
    global_model_test, batch_size=200, shuffle=True)

model_cluster_number = { i:[] for i in range(class_number)}
fed_caps_final_acc=[]
for comunication_iter_index in range(total_comunications):
    # Train
    #if (is_first_comunicate == False):
        #local_training_count = 1

    for _ in range(local_training_count):
        for model_index in range(participant_number):
            print('comminucate round :', model_index)
            data_loader = DataLoader(
                datasets[model_index], batch_size=20, shuffle=True)
            giantCaps_local_models[model_index] = fed_train(giantCaps_local_models[model_index] , data_loader)

    #global_loss, model_cluster_number =  create_test_global_modelv2(fedCaps_local_models, global_test_set_dataloader,  model_config, device, reconstruction_alpha, model_cluster_number, is_first_comunicate)
                                                                               #    is_first_comunicate)

    globalmodel = GlobalFedCaps(conv1_Channel=[model[0]for model in model_configs],
                                primary_channel=[model[1]for model in model_configs],
                                capsule_dimention=[model[2]for model in model_configs],
                                kernel_size=[model[3]for model in model_configs],
                                primaryCaps_filter_size=[model[4]for model in model_configs],
                                routing_iterations=4,
                                n_classes=10,
                                ensimble_num=len(giantCaps_local_models)).to(device)

    globalmodel = create_globalmodel(
        local_models=giantCaps_local_models, global_model=globalmodel)

    globalmodel_loss_val = test(
        "globalmodel", globalmodel, global_test_set_dataloader, reconstruction_alpha)[2]
    print('globalmodel acc iteration  : is  : ',
          globalmodel_loss_val.item())

    fed_caps_final_acc_ensemble.append( globalmodel_loss_val.item())
#
    final_fed_caps_acc =[ test("global model", model  , global_test_set_dataloader)[2] for model in giantCaps_local_models ]

    fed_caps_final_acc.append(final_fed_caps_acc)
    print('final accuracy of fed Caps  model in iteration  : ',
          comunication_iter_index, ' is  : ', final_fed_caps_acc)

torch.save(fed_caps_final_acc, 'final_fed_caps_acc.pth')
torch.save(fed_caps_final_acc_ensemble, 'fed_caps_final_acc_ensemble.pth')


