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
