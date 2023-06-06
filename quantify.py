# python
import threading
import tkinter as tk
from PIL import Image, ImageTk, ImageGrab
import time
import numpy as np
import cv2
from blot_math import Band, CompareBands
import argparse

class KeyTracker():

    def __init__(self, on_key_press, on_key_release, keys : list):
        self.on_key_press = on_key_press
        self.on_key_release = on_key_release
        self.keys = keys
        self.last_press_time = 0
        self.last_release_time = 0

    def track(self, key):
        self.keys.append(key)

    def untrack(self, key):
        self.keys.remove(key)

    def is_pressed(self):
        return time.time() - self.last_press_time < .1

    def report_key_press(self, event):
        if event.keysym in self.keys:
            if not self.is_pressed():
                self.on_key_press(event)
            self.last_press_time = time.time()

    def report_key_release(self, event):
        if event.keysym in self.keys:
            timer = threading.Timer(.1, self.report_key_release_callback, args=[event])
            timer.start()

    def report_key_release_callback(self, event):
        if not self.is_pressed():
            self.on_key_release(event)
        self.last_release_time = time.time()

class MousePositionTracker(tk.Frame):
    """ Tkinter Canvas mouse position widget. """

    def __init__(self, canvas, key_tracker = None):
        self.key_tracker = key_tracker
        self.canvas : tk.Canvas = canvas
        self.canv_width = self.canvas.cget('width')
        self.canv_height = self.canvas.cget('height')
        self.reset()

        # Create canvas cross-hair lines.
        xhair_opts = dict(dash=(3, 2), fill='white', state=tk.HIDDEN)
        self.lines = (self.canvas.create_line(0, 0, 0, self.canv_height, **xhair_opts),
                      self.canvas.create_line(0, 0, self.canv_width,  0, **xhair_opts))

        self.new_ready = True

    def cur_selection(self):
        return (self.start, self.end)

    def begin(self, event):
        self.hide()
        self.start = (event.x, event.y)  # Remember position (no drawing).

    def update(self, event):
        self.end = (event.x, event.y)
        self._update(event) 
        self._command(self.start, (event.x, event.y))  # User callback.

    def _update(self, event):
        # Update cross-hair lines.
        self.canvas.coords(self.lines[0], event.x, 0, event.x, self.canv_height)
        self.canvas.coords(self.lines[1], 0, event.y, self.canv_width, event.y)
        self.show()

    def reset(self):
        self.start = self.end = None

    def hide(self):
        self.canvas.itemconfigure(self.lines[0], state=tk.HIDDEN)
        self.canvas.itemconfigure(self.lines[1], state=tk.HIDDEN)

    def show(self):
        self.canvas.itemconfigure(self.lines[0], state=tk.NORMAL)
        self.canvas.itemconfigure(self.lines[1], state=tk.NORMAL)

    def autodraw(self, command=lambda *args: None, store_command=lambda *args: None, end_command= lambda *args: None):
        """Setup automatic drawing; supports command option"""
        self.reset()
        self._command = command
        self.store_command = store_command
        self.canvas.bind("<Button-1>", self.begin)
        self.canvas.bind("<B1-Motion>", self.update)
        self.canvas.bind("<ButtonRelease-1>", self.quit)
        self.canvas.bind_all('<KeyPress>', self.key_tracker.report_key_press)
        self.canvas.bind_all('<KeyRelease>', self.key_tracker.report_key_release)

    def quit(self, event):
        self.hide()  # Hide cross-hairs.
        self.reset()


class SelectionObject:
    """ Widget to display a rectangular area on given canvas defined by two points
        representing its diagonal.
    """
    def __init__(self, canvas, select_opts, y_range = None):
        # Create attributes needed to display selection.
        self.canvas : tk.Canvas = canvas
        self.select_opts1 = select_opts
        self.width = self.canvas.cget('width')
        self.height = self.canvas.cget('height')
        self.y_range = y_range

        # Options for areas outside rectanglar selection.
        select_opts1 = self.select_opts1.copy()  # Avoid modifying passed argument.
        select_opts1.update(state=tk.HIDDEN)  # Hide initially.
        # Separate options for area inside rectanglar selection.
        select_opts2 = dict(dash=(2, 2), fill='', outline='white', state=tk.HIDDEN)

        # Initial extrema of inner and outer rectangles.
        imin_x, imin_y,  imax_x, imax_y = 0, 0,  1, 1
        omin_x, omin_y,  omax_x, omax_y = 0, 0,  self.width, self.height

        self.rects = (
            # Area *outside* selection (inner) rectangle.
            self.canvas.create_rectangle(omin_x, omin_y,  omax_x, imin_y, **select_opts1),
            self.canvas.create_rectangle(omin_x, imin_y,  imin_x, imax_y, **select_opts1),
            self.canvas.create_rectangle(imax_x, imin_y,  omax_x, imax_y, **select_opts1),
            self.canvas.create_rectangle(omin_x, imax_y,  omax_x, omax_y, **select_opts1),
            # Inner rectangle.
            self.canvas.create_rectangle(imin_x, imin_y,  imax_x, imax_y, **select_opts2)
        )

    def update(self, start, end):
        # Current extrema of inner and outer rectangles.
        imin_x, imin_y,  imax_x, imax_y = self._get_coords(start, end)
        omin_x, omin_y,  omax_x, omax_y = 0, 0,  self.width, self.height

        # Update coords of all rectangles based on these extrema.
        self.canvas.coords(self.rects[0], omin_x, omin_y,  omax_x, imin_y),
        self.canvas.coords(self.rects[1], omin_x, imin_y,  imin_x, imax_y),
        self.canvas.coords(self.rects[2], imax_x, imin_y,  omax_x, imax_y),
        self.canvas.coords(self.rects[3], omin_x, imax_y,  omax_x, omax_y),
        self.canvas.coords(self.rects[4], imin_x, imin_y,  imax_x, imax_y),

        for rect in self.rects:  # Make sure all are now visible.
            self.canvas.itemconfigure(rect, state=tk.NORMAL)

    def _get_coords(self, start, end):
        """ Determine coords of a polygon defined by the start and
            end points one of the diagonals of a rectangular area.
        """
        return (min((start[0], end[0])), min((start[1], end[1])),
                max((start[0], end[0])), max((start[1], end[1])))

    def hide(self):
        for rect in self.rects:
            self.canvas.itemconfigure(rect, state=tk.HIDDEN)

    def get_inner_rect(self):
        return self.rects[-1]
    
    def get_width_height_inner_rect(self):
        tmp = self.canvas.coords(self.get_inner_rect())
        return tmp[2] - tmp[0], tmp[3] - tmp[1]
    
class FixedSizeObject(SelectionObject):

    def __init__(self, canvas, select_opts, width, height, y_range = None):
        super().__init__(canvas, select_opts, y_range)
        self.fixed_width = width
        self.fixed_height = height

    def update(self, start, end):
        # Current extrema of inner and outer rectangles.
        imin_x, imin_y,  imax_x, imax_y = self._get_coords(start, end)
        omin_x, omin_y,  omax_x, omax_y = 0, 0,  self.width, self.height

        # Update coords of all rectangles based on these extrema.
        self.canvas.coords(self.rects[0], omin_x, omin_y,  omax_x, imin_y),
        self.canvas.coords(self.rects[1], omin_x, imin_y,  imin_x, imax_y),
        self.canvas.coords(self.rects[2], imax_x, imin_y,  omax_x, imax_y),
        self.canvas.coords(self.rects[3], omin_x, imax_y,  omax_x, omax_y),
        self.canvas.coords(self.rects[4], imin_x, imin_y,  imax_x, imax_y),

        for rect in self.rects:  # Make sure all are now visible.
            self.canvas.itemconfigure(rect, state=tk.NORMAL)

    def _get_coords(self, start, end):
        return end[0] - self.fixed_width/2, end[1] - self.fixed_height/2,\
               end[0] + self.fixed_width/2, end[1] + self.fixed_height/2

class SelectedObjects:
    """
    Widget to store selected rectangles for further processing
    """

    def __init__(self, canvas, select_opts = None):
        if select_opts is None:
            select_opts = dict(dash=(2, 2), stipple='gray25', fill='',
                          outline='')
        self.canvas : tk.Canvas = canvas
        self.objs = None
        self.initialized = False

    def __iter__(self):
        return iter(self.objs)

    def initialize(self):
        self.objs : list[SelectionObject] = []
        self.initialized = True

    def new_selelction_object(self, canvas, select_opts, key):
        if key == 'startup':
            r = SelectionObject(canvas, select_opts)
        elif key == 'b':
            r = SelectionObject(canvas, select_opts)
            self.remove_last_rect()
        elif key == 'space':
            r = FixedSizeObject(canvas, select_opts, *self.objs[-1].get_width_height_inner_rect())
        else:
            raise ValueError(f"<key> : {key} -  is not a valid key. Known keys are <space> and <b>")
        self.add_rect(r)
        return r

    def add_rect(self, selection_object : SelectionObject):
        assert self.initialized, 'Initialize first'
        rect = selection_object.get_inner_rect()
        self.objs.append(selection_object)
        self.canvas.itemconfigure(rect, state=tk.NORMAL)

    def remove_last_rect(self):
        if self.initialized:
            rect = self.objs[-1].get_inner_rect()
            self.canvas
            self.canvas.itemconfigure(rect, state=tk.HIDDEN)
            del self.objs[-1]

    def end(self):
        for obj in self.objs:
            obj.hide()
        del self.objs
        self.initialized = False
        self.fixed_size_flag = False

class DummySelectionObject:
    """
    Placeholder for when all objects have been created
    """

    def __init__(self, *args):
        pass

    def update(self, *args):
        pass

    def hide(self, *args):
        pass

def fancy_math(rects):
    from matplotlib import pyplot as plt
    bands = []
    for rect in rects:
        bands.append(Band(rect))
    for band in bands:
        band : Band
        plt.plot(band.plot_curve())
    return 0
    print(sum(rects))

class Application(tk.Frame):

    # Default selection object options.
    SELECT_OPTS = dict(dash=(2, 2), stipple='gray25', fill='',
                          outline='')

    def __init__(self, parent, nr_objs, image_path, beautify_image= False, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.nr_objs = nr_objs
        path = image_path
        if not beautify_image:
            img_PIL = Image.open(path).convert('L')
        else:
            # read the image
            img = cv2.imread(path)
            # convert to gray
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # blur
            smooth = cv2.GaussianBlur(gray, (95,95), 0)
            # divide gray by morphology image
            division = cv2.divide(gray, smooth, scale=192)

            img_PIL = Image.fromarray(np.uint8(division), mode= 'L')
            
        img = ImageTk.PhotoImage(img_PIL)
        self.canvas = tk.Canvas(root, width=img.width(), height=img.height(),
                                borderwidth=0, highlightthickness=0)
        self.canvas.pack(expand=True)

        self.canvas.create_image(0, 0, image=img, anchor=tk.NW)
        self.canvas.img = img  # Keep reference.
        self.canvas.img_array = np.asarray(img_PIL)

        # Create selection object to show current selection boundaries.
        self.selection_objs = self.create_selection_objs()
        self.selection_obj = self.get_next_selection_obj('startup')
        self.bands = CompareBands()
        self.background = 0


        # Callback function to update it given two points of its diagonal.
        def on_drag(start, end, **kwarg):  # Must accept these arguments.
                self.selection_obj.update(start, end)
        
        def on_press(event):
            key = event.keysym
            if key == 'b':
                bbox = np.array(self.canvas.coords(self.selection_obj.get_inner_rect()), dtype=int)
                self.background = 255 - self.canvas.img_array[bbox[1]:bbox[3], bbox[0]:bbox[2]].mean()
                self.selection_obj = self.get_next_selection_obj(key)
            elif key == 'space':
                bbox = np.array(self.canvas.coords(self.selection_obj.get_inner_rect()), dtype=int)
                self.bands.add_band(self.canvas.img_array[bbox[1]:bbox[3], bbox[0]:bbox[2]], bbox[0])
                self.selection_objs.fixed_size_flag = True
                self.selection_obj = self.get_next_selection_obj(key)
            elif key == 'Return':
                # if type(self.selection_obj) is not DummySelectionObject:
                #     bbox = np.array(self.canvas.coords(self.selection_obj.get_inner_rect()), dtype=int)
                #     self.bands.add_band(self.canvas.img_array[bbox[1]:bbox[3], bbox[0]:bbox[2]])
                self.bands.math(background= self.background)


        def on_release(event):
            if event.keysym == 'q':
                self.bands.reset()
                self.selection_objs.end()
                del self.selection_objs
                self.selection_objs = self.create_selection_objs()
                self.selection_obj = self.get_next_selection_obj('startup')

        key_tracker = KeyTracker(on_press, on_release, ['space', 'Return', 'q', 'b'])
        # Create mouse position tracker that uses the function.
        self.posn_tracker = MousePositionTracker(self.canvas, key_tracker)
        self.posn_tracker.autodraw(command=on_drag)  # Enable callbacks.

    # def create_selection_objs(self):
    #     l = [SelectionObject(self.canvas, self.SELECT_OPTS) for _ in range(self.nr_objs)]
    #     l.append(DummySelectionObject())
    #     self.selection_objs_iter = iter(range(len(l)))
    #     return l

    def create_selection_objs(self):
        objs = SelectedObjects(self.canvas, self.SELECT_OPTS)
        objs.initialize()
        return objs

    # def get_next_selection_obj(self):
    #     return self.selection_objs[next(self.selection_objs_iter)]

    def get_next_selection_obj(self, key):
        return self.selection_objs.new_selelction_object(self.canvas, self.SELECT_OPTS, key)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', type=str, help='Path to image file. (With extension pls)')
    parser.add_argument('-p', '--preprocess', action='store_true', help='If present image is preprocessed however the fuck i feel like')
    args = parser.parse_args()

    WIDTH, HEIGHT = 900, 900
    BACKGROUND = 'grey'
    TITLE = 'EZ Western Blot Quantification'

    root = tk.Tk()
    root.title(TITLE)
    root.geometry('%sx%s' % (WIDTH, HEIGHT))
    root.configure(background=BACKGROUND)

    app = Application(root, 3, args.img_path, args.preprocess, background=BACKGROUND)
    app.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.TRUE)
    app.mainloop()