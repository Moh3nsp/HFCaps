import math

import matplotlib.pyplot as plt
import torch
from statistics import mean

#plt.figure(dpi=1000)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

giant_model = torch.load('fed_caps_giant_800.pth',map_location=device)
fed_caps_models = torch.load('fed_caps_models.pth',map_location=device)

highest_in_all_models = [max(model_acc) for model_acc in fed_caps_models]
lowest_in_all_models = [min(model_acc) for model_acc in fed_caps_models]
average_in_all_models = [mean([int(arr) for arr in model_acc ])  for model_acc in fed_caps_models]

#
# for index in range(10):
#     weak_learner=[model_acc[index] for model_acc in fed_caps_models]
#     plt.plot([j for j in range(len(weak_learner))], weak_learner, '-', markerfacecolor="None",
#              c='0.7', alpha=0.6,zorder=1)
#
# plt.plot([ i for i in range(len(giant_model))] , giant_model, '-', markerfacecolor="",
#                  label='FedCAps', c='r',zorder=9)
#
# plt.plot([ i for i in range(len(highest_in_all_models))] , highest_in_all_models, '-', markerfacecolor="None",
#                label='FedCAps', c='b',zorder=2)
#
# plt.plot([ i for i in range(len(lowest_in_all_models))] , lowest_in_all_models, '-', markerfacecolor="None",
#                   label='FedCAps', c='y',zorder=3)
#
# plt.plot([ i for i in range(len(average_in_all_models))] , average_in_all_models, '-', markerfacecolor="None",
#                  label='FedCAps', c='g',zorder=4)


plt.title(f'Learning curve ')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.legend( framealpha=0.01)
plt.show()
print('lowest : min-dif' , (giant_model[0] - highest_in_all_models[0]))
print('lowest : max-dif' ,max( giant_model ) - max(highest_in_all_models))