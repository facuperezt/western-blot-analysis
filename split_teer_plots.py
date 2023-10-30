#%%
import numpy as np
import pandas as pd
from typing import List, Dict
from matplotlib import pyplot as plt
# %%
def split_among_airlift(data_frame: pd.DataFrame) -> List[pd.DataFrame]:
    x_axis = data_frame.loc[:, "Tag"].to_numpy()
    airlift_cols = [col for col in data_frame.columns if "Airlift" in col]
    non_airlift_cols = data_frame.columns.difference(airlift_cols)
    return {
        type: data_frame.loc[:, cols] for type, cols 
        in zip(["Normal", "Airlift"],[non_airlift_cols, airlift_cols])
        }

def split_among_experiments(
        data_frame: pd.DataFrame,
        known_experiments: List[str] = ["TNF", "1:5", "1:10"],
        ) -> List[pd.DataFrame]:
    out = []
    used_columns = ["Tag", "Wachstumsmedium"]
    known_experiments.append(known_experiments.pop(known_experiments.index("TNF")))
    for experiment in known_experiments:
        experiment_columns = []
        for col in data_frame.columns.difference(used_columns):
            if experiment in col or experiment == "TNF":
                experiment_columns.append(col)
        out.append(data_frame.loc[:, experiment_columns])
        used_columns.extend(experiment_columns)

    return {key: value for key, value in zip(known_experiments[::-1], out[::-1])}

def get_experiments(file_path: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    df = pd.read_excel(file_path)
    x_axis = df.loc[:, "Tag"].to_numpy()
    without_and_with = split_among_airlift(df)
    without_and_with
    all_experiments = {
        type: split_among_experiments(_df) for type, _df
        in without_and_with.items()
        }
    blank = df.loc[:, "Wachstumsmedium"].to_numpy()
    return x_axis, all_experiments, blank

def plot_arrows_in_teer(ax, stimulationen, data):
    plt.sca(ax)
    y_lims = ax.get_ylim()
    arrow_start = -50
    arrow_len = 150
    head_length = 0.25*arrow_len
    plt.vlines(stimulationen, [-10,-10,-10], [1500, 1500, 1500], ['gray']*3, linestyles=['dashed']*3, alpha=[0.5]*3, label= 'Stimulation')
    plt.arrow(stimulationen[0], arrow_start, 0, arrow_len, width= 0.01, head_length= head_length, head_width= 0.3, length_includes_head= True, label= 'Stimulation', color="black")
    plt.arrow(stimulationen[1], arrow_start, 0, arrow_len, width= 0.01, head_length= head_length, head_width= 0.3, length_includes_head= True, label= 'Stimulation', color="black")
    plt.arrow(stimulationen[2], arrow_start, 0, arrow_len, width= 0.01, head_length= head_length, head_width= 0.3, length_includes_head= True, label= 'Stimulation', color="black")
    plt.gca().set_ylim(y_lims)

def plot_single_experiment(ax: plt.Axes, x_axis: np.ndarray, data: pd.DataFrame, blank: np.ndarray = None, stimulationen = None, savefig_name= "", labels=None):
    if stimulationen is None:
        stimulationen = [6, 7, 8]
    else:
        stimulationen = [int(t) for t in stimulationen]
    if savefig_name == "": # for consistency with GUI
        savefig_name = None
    if np.isnan(data).any().any(): missing_data = True
    else: missing_data = False
    legend = data.columns.tolist()
    data = data.to_numpy()
    lines = []
    for c, style, line, label in zip(["gray", "black"], ['-o', '-D'], data.transpose()[::-1], labels):
        plt.sca(ax)
        lines.append(plt.plot(x_axis, line, style, color=c, alpha=0.9, label=label, markeredgewidth=0)[0])
    ax.plot(x_axis, blank, ":", label="Wachstumsmedium", color="gray", alpha= 0.3)
    if missing_data:
        missing_days, missing_curves = np.where(np.isnan(data))
        data_completed = data.copy()
        for day in missing_days: # slow but versatile
            if day == 0 or day == data.shape[0]: continue
            for curve in missing_curves:
                data_completed[day, curve] = (data_completed[day-1, curve] + data_completed[day+1, curve])/2
        ax.plot(x_axis, data_completed, ls='dashed')
    plot_arrows_in_teer(ax, stimulationen, data)
    # ax.legend(legend, frameon=False)
    ax.set_xticks(x_axis)
    ax.set_xticklabels([_x if _x % 5 == 0 else None for _x in x_axis])
    # ax.set_xlabel('Kultivierungsdauer nach der Calcium-Umstellung [d]')
    ax.set_ylabel(r'TEER [$\Omega$ x $cm^2$]')
    ax.set_ylim(data.min()-(data.max()*1.1 - data.max()), data.max()*1.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if savefig_name is not None: plt.savefig(savefig_name)
    return lines

# %%
file_path = "TEER_Python_MW.xlsx"
x_axis, all_experiments, blank = get_experiments(file_path)
fig, axs = plt.subplots(2, 3, figsize=(16,8), sharex=False, sharey=True)
letters = (letter for letter in "ABCDEFGHIJKLMNO")
mapping = {}
legend_mapping = {
    "TNF": "TNF\u03B1",
    "1:10": "M1 (1:10)",
    "1:5": "M1 (1:5)"
}
for axs_row,(name, airlift) in zip(axs, all_experiments.items()):
    for ax, (experiment_name, experiment) in zip(axs_row, airlift.items()):
        ax: plt.Axes
        lines = plot_single_experiment(ax, x_axis, experiment, blank, labels=experiment_name)
        print(lines)
        mapping[next(letters)] = f"{name} - {experiment_name}"
        ax.legend(lines, ["Stimuliert", legend_mapping[experiment_name]])
        ax.set_ylim(-50, 1100)
        ax.yaxis.set_tick_params(which="major", labelleft=True)

letters = (letter for letter in "ABCDEFGHIJKLMNO")
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.6)
for ax in axs.flatten():
    ax.text(-5, ax.get_ylim()[1]*1.1, next(letters), fontdict={"size":21, "weight": "bold"})
fig.supxlabel(f'Kultivierungsdauer nach der Calcium-Umstellung [d]\n')
plt.suptitle(mapping)
fig.savefig("output/all_teer_plots.pdf")

for name, airlift in all_experiments.items():
    for experiment_name, experiment in airlift.items():
        fig = plt.figure()
        ax = plt.gca()
        ax: plt.Axes
        lines = plot_single_experiment(ax, x_axis, experiment, blank, labels=experiment_name)
        print(lines)
        # mapping[next(letters)] = f"{name} - {experiment_name}"
        ax.legend(lines, ["Kontrolle", legend_mapping[experiment_name]])
        ax.set_ylim(-50, 1100)
        ax.yaxis.set_tick_params(which="major", labelleft=True)
        ax.set_xlabel(f'Kultivierungsdauer nach der Calcium-Umstellung [d]\n')
        ax.set_title(name)
        fig.savefig(f"output/{name}_{legend_mapping[experiment_name]}.pdf")
# %%
