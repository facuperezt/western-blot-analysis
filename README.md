# western-blot-analysis
Quick project because ImageJ doesn't port very well to Mac

## Automatic Detection
``` bash
python auto_band.py --image_path <image_path>
```

For now it will just show the detected bands and plot the quantified values. To control the band detection manually use quantify.py

## Manual Detection
``` bash
python quantify.py <image_path>
```

Inside the program mark a rectangle around the biggest band and confirm with spacebar. Afterwards the program will remember the size of the mark for better results.
When all bands are marked press Enter to plot the curves and compute the area under the curve.

### Extra options
- 'q' -  Resets view.
- 'b' -  Mark a piece of the background and press 'b', the plots will substract it from the end result. (unstable)

#### Future features?
- change image without closing program
- export values directly to excel
