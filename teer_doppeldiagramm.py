# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

data = pd.read_excel("TEER_Python_MW.xlsx", index_col="Tag")

# %%
x = data.index
names = data.columns
werte = data.to_numpy()

# xxx.shape --> Dimension der Variablen überprüfen
# %%
# große TEER-Abbildung mit allen Versuchen

def teer_plot(x, werte, names):
    plt.figure(figsize=(10, 6))
    plt.title("TEER")
    plt.plot(x, werte, label=names)
    plt.legend(bbox_to_anchor=(1.025, 1.0), loc='upper left')
    plt.xlabel("Kultivierungsdauer nach der Calciumumstellung [d]")
    plt.ylabel("TEER [\u03A9 x cm$^{2}$]")
    plt.xticks(x)
    plt.ylim(0, 1050)
    
    plt.arrow(6, 35, 0, 100, width=.04, color='black', head_length=14)
    plt.arrow(7, 35, 0, 100, width=.04, color='black', head_length=14)
    plt.arrow(8, 35, 0, 100, width=.04, color='black', head_length=14)

teer_plot(x, werte, names)

# %%
# zwei TEER-Abbildungen mit 2 Legenden

fig, axs = plt.subplots(1, 2, figsize=(16, 6))

def teer_plot2(x, werte, names, ax:plt.Axes, title):
    plt.sca(ax)     # sca = set current axis
    plt.plot(x, werte, label=names)
    ax.set_title(title)
    plt.legend(loc='upper left')
    plt.xlabel("Kultivierungsdauer nach der Calciumumstellung [d]")
    plt.ylabel("TEER [\u03A9 x cm$^{2}$]")
    plt.xticks(x)
    plt.ylim(0, 1050)
    
    plt.arrow(6, 35, 0, 100, width=.04, color='black', head_length=14)
    plt.arrow(7, 35, 0, 100, width=.04, color='black', head_length=14)
    plt.arrow(8, 35, 0, 100, width=.04, color='black', head_length=14)

teer_plot2(x, werte[:, 1:7], names[1:7], axs[0], "Ohne Airlift")
axs[0].plot(x, werte[:, 0], label=names[0], color='gray')
axs[0].legend(loc='upper left')
teer_plot2(x, werte[:, 7:], names[7:], axs[1], "Mit Airlift")

# fig.suptitle("TEER")

# %%
# TEER-Abbildung mit und ohne Airlift mit einer Legende rechts

fig, axs = plt.subplots(1, 2, figsize=(18, 5.5))

def teer_plot2(x, werte, names, ax:plt.Axes, title):
    plt.sca(ax)     # sca = set current axis
    plt.plot(x, werte, label=names, marker="^")
    ax.set_title(title, fontdict={"size":18})

    plt.xticks(x-0.5)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.xaxis.set_minor_locator(ticker.FixedLocator(x))
    ax.xaxis.set_minor_formatter(ticker.FixedFormatter(x))
    ax.tick_params(which="minor", bottom=False, labelsize=12)
    ax.tick_params("y", which="major", labelsize=12)
    
    plt.xlabel("Kultivierungsdauer nach der Calciumumstellung [d]", fontsize=16)
    plt.ylabel("TEER [\u03A9 x cm$^{2}$]", fontsize=16)
    plt.xlim(0.5, 15.5)
    plt.ylim(0, 1050)
    
    plt.arrow(6, 35, 0, 100, width=.04, color='black', head_length=14)
    plt.arrow(7, 35, 0, 100, width=.04, color='black', head_length=14)
    plt.arrow(8, 35, 0, 100, width=.04, color='black', head_length=14)


teer_plot2(x, werte[:, 1:7], names[1:7], axs[0], "Ohne Airlift")
axs[0].plot(x, werte[:, 0], label=names[0], color='gray', marker=".")
axs[0].legend(bbox_to_anchor=(2.2, 1.0), loc='upper left', fontsize=16)
teer_plot2(x, werte[:, 7:], names[7:], axs[1], "Mit Airlift")


# %%
