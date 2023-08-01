#%%
#!python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import glob
from matplotlib.patches import Patch
from matplotlib.path import Path

def expand_index(idx):
    new_idx = idx.copy() * 2
    return np.stack([new_idx, new_idx+1])

def process_data(data, indexes, protein_filter= None, clip= True):
    _, index = np.unique(indexes.flatten(), return_index= True)
    x_axis_tick_labels =  indexes.flatten()[sorted(index)]
    y_values = []
    for thing in x_axis_tick_labels:
        idx_x, idx_y = np.where(indexes == thing)
        idx_x = expand_index(idx_x)
        mean_value = np.mean(data[idx_x, idx_y])
        y_values.append(mean_value)

    y_values = np.array(y_values)
    order = np.argsort(y_values)

    neg_mean = y_values[np.where(x_axis_tick_labels == 'Neg')]
    pos_mean = y_values[np.where(x_axis_tick_labels == 'Pos')]

    y_values = y_values - neg_mean
    if clip:
        y_values = np.clip(y_values, 0, None)
    y_values = y_values / pos_mean

    if protein_filter is not None:
        included = np.array([i for i, protein in enumerate(x_axis_tick_labels) if protein in protein_filter])
        x_axis_tick_labels = x_axis_tick_labels[included]
        y_values = y_values[included]
        order = np.argsort(y_values)

    return x_axis_tick_labels, y_values, order

def split_pro_anti_inf(all_proteins, values, pro_inf, anti_inf):
    pro = []
    anti = []
    rest = []
    for protein, value in zip(all_proteins, values):
        if protein in pro_inf:
            pro.append([protein, value])
        elif protein in anti_inf:
            anti.append([protein, value])
        else:
            rest.append([protein, value])
    return pro, anti, rest

def filter_by_diff(y_diff, increase=2, decrease=2):
    increases = np.where(y_diff >= increase)
    decreases = np.where(y_diff <= -1/decrease)

    return np.sort(np.hstack([increases, decreases]).squeeze())


files = glob.glob("./data/*") # Get all files from /data/
indexes = np.array([
    ['Pos', 'Pos', 'Neg', 'Neg', 'ENA-78', 'GCSF', 'GM-CSF', 'GRO', 'GRO-a', 'I-309', 'IL-1a', 'IL-1b'],
    ['IL-2', 'IL-3', 'IL-4', 'IL-5', 'IL-6', 'IL-7', 'IL-8', 'IL-10', 'IL-12', 'IL-13', 'IL-15', 'INF-g'],
    ['MCP-1', 'MCP-2', 'MCP-3', 'MCSF', 'MDC', 'MIG', 'MIP-1 d', 'RANTES', 'SCF', 'SDF-1', 'TARC', 'IGF-b1'],
    ['TNF-a', 'TNF-b', 'EGF', 'IGF-I', 'Angiogenin', 'Oncostatin M', 'Thrombopoietin', 'VEGF', 'PDGF BB', 'Leptin', 'Neg', 'Pos'],
])
_, index = np.unique(indexes.flatten(), return_index= True)
index_order =  indexes.flatten()[sorted(index)]
pro_inf = np.array(['TNF-a', 'MCP-1', 'IL-5'])
anti_inf = np.array(['TNF-b', 'ENA-78'])
rest = np.array([prot for prot in index_order if prot not in pro_inf and prot not in anti_inf])
#%%
tables = []
for file in files:
    print(file)
    info = pd.read_csv(file, delimiter=";", encoding="utf-8") # Read from file
    info = np.array(info)[:, 1:] # Make to array and drop index
    tables.append(info) # Append to list

data_vor = tables[1]
data_nach = tables[0]

pfilter = None
x, y_vor, order_vor = process_data(data_vor, indexes, pfilter)
x_nach, y_nach, order_nach = process_data(data_nach, indexes, pfilter)

y_diff = np.divide(y_nach - y_vor, y_vor, out= np.zeros_like(y_vor), where=y_vor != 0)

filt = filter_by_diff(y_diff, increase=2, decrease=2)


x = x[filt]
y_vor = y_vor[filt]
y_nach = y_nach[filt]
y_diff = y_diff[filt]

fig, ax = plt.subplots(1,1, figsize=(9,6))
if False:

    order_plot = np.arange(len(x))
    order_plot = order_plot[y_nach < 2]

    x = x[order_plot]
    y_vor = y_vor[order_plot]
    y_nach = y_nach[order_plot]
    y_diff = y_diff[order_plot]
    ax.set_ylim(0, 1.75)
else:
    ax.set_ylim(0, 11.75)


colors_dict = {-1 : 'r', 1 : 'g'}

# plt.style.use('grayscale')
# ax.grid(True, which= 'both', axis='y')
# ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
bar_nach = ax.bar(x, y_nach, 0.7, alpha=0.7, label='Nach der Aktivierung')
bar_vor = ax.bar(x, y_vor, 0.35, alpha=0.7, label='Vor der Aktivierung')
bar_diff = ax.bar(x, y_nach - y_vor, 0.05, y_vor, color=[colors_dict[np.sign(a)] for a in y_diff], label='Differenz')
ax.set_xticklabels(x, rotation= 35)
ax.set_ylabel('Normalized pixel density [-]')


legend_elements = [bar_nach, bar_vor, 
                   Patch(facecolor='red', edgecolor='green', linewidth= 3, label= 'Differenz')]
ax.legend(loc = 'best', handles= legend_elements, numpoints=10, frameon=False)
fig.patch.set_facecolor('white')
# fig.savefig("cytokin_array.svg")
# plt.show()
# %%
