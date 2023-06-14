import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.metrics import auc

def scale_array(x, new_size):
    return np.asarray(Image.fromarray(x).resize((new_size[1], new_size[0])))

class Band:
    """
    Assumes bands are bigger horizontally than vertically
    """
    def __init__(self, arr, normalized_size = (1000, 100), normalized = False):
        self.arr : np.ndarray = arr
        self.nsize = normalized_size
        self._normalized = normalized

    def __eq__(self, obj, other):
        return obj.arr == other.arr

    def _normalize(self):
        if self._normalized and self.arr.shape == self.nsize:
            pass
        else:
            if self.arr.shape[0] < self.arr.shape[1]:
                self.arr = self.arr.T
            self.arr = scale_array(self.arr, self.nsize)
            self.arr = np.ones_like(self.arr) - self.arr/255
            self._normalized = True

    def intensity_curve(self) -> np.ndarray:
        if not self._normalized:
            self._normalize()
        mean = self.arr.mean(axis=0)

        # Remove background again

        return mean - np.linspace(mean[0], mean[-1], 100)

    def plot_curve(self, ax : plt.Axes, background = 0):
        crv = self.intensity_curve()
        x = np.linspace(0,1,len(crv))
        ax.plot(x, crv - background)
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.hlines(0, -1, 3, 'gray', 'dashed', alpha= 0.4)
        ax.vlines(x[np.argmax(crv)], -1, 3, 'gray', 'dashed', alpha= 0.4)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_title(self.quantify_peak())
        ax.axis('off')

    def plot_bar(self, ax : plt.Axes):
        peak = self.quantify_peak()
        ax.bar(0, peak)
        ax.set_title(f"{peak:.4f}")
        ax.axis('off')


    def quantify_peak(self):
        crv = self.intensity_curve()
        # x = np.linspace(0,1,len(crv))

        if crv.argmax() != 0 and crv.argmax() != len(crv) and False:
            inds = [np.abs(np.diff(c, 2)).argmin() for c in [crv[:crv.argmax()], crv[crv.argmax():]]]
            crv = crv[inds[0]:inds[1] + crv.argmax()]

        return auc(np.linspace(0,1,len(crv)), crv)







class CompareBands:
    """
    Band container
    """
        
    def __init__(self):
        self.bands : list[Band] = []
        self.leftmost_points : list[int] = [] # contains the leftmost pixel of each band
        self._ready_to_quantify = False

    def add_band(self, band, left_pixel):
        self.bands.append(Band(band))
        self.leftmost_points.append(left_pixel)
        order = np.argsort(self.leftmost_points)
        self.bands = [self.bands[i] for i in order]
        self.leftmost_points = [self.leftmost_points[i] for i in order]

    def rm_band(self, band):
        idx = self.bands.index(Band(band))
        self.bands.pop(idx)
        self.leftmost_points.pop(idx)

    def plot_bands(self, background = 0):
        fig, axs = plt.subplots(len(self.bands), 1)
        for ax, band in zip(np.asarray(axs), self.bands):
            band.plot_curve(ax, background)
        fig.show()


    def math(self, background= 0):
        fig, axs = plt.subplots(1, len(self.bands), sharex= True, sharey= True)
        self.plot_bars(axs, background)
        self._ready_to_quantify = True
        fig.show()

    def plot_bars(self, axs, background= 0):
        for ax, band in zip(np.asarray(axs), self.bands):
            band.plot_bar(ax)

    def reset(self):
        del self.bands
        self.bands : list[Band] = []
        self.leftmost_points : list[int] = []
        self._ready_to_quantify = False

