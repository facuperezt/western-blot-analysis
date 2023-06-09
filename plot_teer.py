#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help='Name von die Datei.')
    parser.add_argument('-s', '--stimulation_days', type= int, nargs='*', required= False, help='Tagen an den Stimuliert wurde (default ist 7 8 9)')
    parser.add_argument('--save_name', type=str, required= False, help='Name von die plot datei die erstellt wird. Wenn man kein Name gibt wird es nicht gespeichert, nur angezeigt.')

    return parser.parse_args()

def plot_differences(data_path = None, stimulationen = None, savefig_name = ""):
    if data_path is None:
        raise ValueError('Schatzi die Dateiname nicht vergessen.')
    if stimulationen is None:
        stimulationen = [7, 8, 9]
    else:
        stimulationen = [int(t) for t in stimulationen]
    if savefig_name == "": # for consistency with GUI
        savefig_name = None
    data = pd.read_excel(data_path).to_numpy()
    legend = data[1:, 0]
    x_ticks = data[0, 1:].astype(int)
    data = data[1:, 1:].T.astype(np.float64)
    stim = data[stimulationen[0]-1:stimulationen[-1]+1]
    stim_diff = stim[1:] - stim[:-1]
    legend = legend[1:]
    stim_diff = stim_diff[:,1:]

    fig, ax = plt.subplots(figsize=(8,5))
    x = np.arange(stim_diff.shape[1])

    for i, day in enumerate(stim_diff):
        p = ax.plot(x, day, '--o', label=f"Day: {stimulationen[i]}{chr(int('2192', 16))}{stimulationen[i]+1}")

    ax.set_xticks(x)
    ax.set_xticklabels(legend, rotation=30)
    ax.set_ylabel(r"$\Delta$ pro Tag [$\Omega$ x $cm^2$]", fontdict={'size': 15})
    ax.set_xlabel(r"Cytokine", fontdict={'size': 15})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(visible=True, which='major', axis='y')
    ax.set_axisbelow(True)
    ax.legend(frameon=True)
    ax.set_title("Pro-Tag Differenz über die Stimulationstage")
    fig.show()

def plot_total_differences(data_path = None, stimulationen = None, savefig_name = ""):
    if data_path is None:
        raise ValueError('Schatzi die Dateiname nicht vergessen.')
    if stimulationen is None:
        stimulationen = [7, 8, 9]
    else:
        stimulationen = [int(t) for t in stimulationen]
    if savefig_name == "": # for consistency with GUI
        savefig_name = None
    data = pd.read_excel(data_path).to_numpy()
    legend = data[1:, 0]
    x_ticks = data[0, 1:].astype(int)
    data = data[1:, 1:].T.astype(np.float64)
    stim = data[stimulationen[0]-1:stimulationen[-1]+1]
    stim_diff = stim[1:] - stim[:-1]
    stim_diff_total = stim_diff.sum(axis=0)[1:]

    fig, ax = plt.subplots(figsize=(9,6))
    colors = {-1 : 'red', 1 : 'blue'}
    x = np.arange(len(stim_diff_total))
    ax.bar(x,stim_diff_total, width= 0.6, alpha= 0.7, color= [colors[np.sign(val)] for val in stim_diff_total])
    x_lims = ax.get_xlim()
    ax.hlines(stim_diff_total[0], -10, 20, alpha= 0.4, linestyles='dashed', colors= 'gray')
    ax.set_xlim(x_lims)
    ax.set_xticks(x)
    ax.set_xticklabels(legend[1:], rotation=40)
    ax.set_ylabel(r"$\Delta$ über 3 Tage [$\Omega$ x $cm^2$]", fontdict={'size': 15})
    ax.set_xlabel(r"Cytokine", fontdict={'size': 15})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.show()

def plot_teer(data_path = None, stimulationen = None, savefig_name= ""):
    if data_path is None:
        raise ValueError('Schatzi die Dateiname nicht vergessen.')
    if stimulationen is None:
        stimulationen = [7, 8, 9]
    else:
        stimulationen = [int(t) for t in stimulationen]
    if savefig_name == "": # for consistency with GUI
        savefig_name = None
    data = pd.read_excel(data_path).to_numpy()
    legend = data[1:, 0]
    x_ticks = data[0, 1:].astype(int)
    data = data[1:, 1:].T.astype(np.float64)
    if np.isnan(data).any(): missing_data = True
    else: missing_data = False
    fig = plt.figure(figsize=(9,6))
    plt.plot(x_ticks, data)
    if missing_data:
        missing_days, missing_curves = np.where(np.isnan(data))
        data_completed = data.copy()
        for day in missing_days: # slow but versatile
            if day == 0 or day == data.shape[0]: continue
            for curve in missing_curves:
                data_completed[day, curve] = (data_completed[day-1, curve] + data_completed[day+1, curve])/2
        plt.plot(x_ticks, data_completed, ls='dashed')
    ax = plt.gca()
    y_lims = ax.get_ylim()
    plt.arrow(stimulationen[0], -100, 0, 60, width= 0.01, head_length= 20, head_width= 0.18)
    plt.arrow(stimulationen[1], -100, 0, 60, width= 0.01, head_length= 20, head_width= 0.18)
    plt.arrow(stimulationen[2], -100, 0, 60, width= 0.01, head_length= 20, head_width= 0.18, label= 'Stimulation')
    plt.vlines(stimulationen, [-10,-10,-10], [1500, 1500, 1500], ['gray']*3, linestyles=['dashed']*3, alpha=[0.5]*3, label= 'Stimulation')
    plt.gca().set_ylim(y_lims)
    plt.legend(legend, frameon= False)
    plt.xticks(x_ticks)
    plt.xlabel('Kultivierungsdauer nach der Calcium-Umstellung [d]')
    plt.ylabel(r'TEER [$\Omega$ x $cm^2$]')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()
    if savefig_name is not None: plt.savefig(savefig_name)


if __name__ == '__main__':
    args = parse_args()
    plot_teer(args.file_path, args.stimulation_days, args.save_name)
# %%
