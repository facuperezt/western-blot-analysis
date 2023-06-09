#!/usr/bin/env python
import tkinter as tk
from plot_teer import plot_teer, plot_differences, plot_total_differences

def process_inputs():
    string_input = entry_string.get()
    list_input = entry_list.get().split(",")
    save_input = entry_string2.get()
    
    # Call your desired function here, passing the inputs
    fn = available_plots[current_plot.get()]
    fn(string_input, list_input, save_input)
    
    # Clear the input fields
    # entry_string.delete(0, tk.END)
    # entry_list.delete(0, tk.END)
    entry_string2.delete(0, tk.END)

if __name__ == '__main__':
    # Create the main window
    window = tk.Tk()
    window.title("Input GUI")

    available_plots = {
        'Plot über Tage' : plot_teer,
        'Plot Differenz bei Stim' : plot_differences,
        'Plot Gesamtdifferenz bei Stim' : plot_total_differences,
    }

    current_plot = tk.StringVar()
    current_plot.set(list(available_plots.keys())[0])
    # Create Dropdown menu
    drop = tk.OptionMenu(window, current_plot, *available_plots.keys() )
    drop.pack()

    # String Input
    label_string = tk.Label(window, text="Datei Name:")
    label_string.pack()
    entry_string = tk.Entry(window)
    entry_string.pack()
    entry_string.insert(0, "TEER.xlsx")

    # List Input
    label_list = tk.Label(window, text="Stimulationstage (comma-separated):")
    label_list.pack()
    entry_list = tk.Entry(window)
    entry_list.pack()
    entry_list.insert(0, "7,8,9")

    # Boolean Input
    save_name = tk.Label(window, text="Plot Speichern Unter: (Falls man nichts eingibt wird es nicht gespeichert)")
    save_name.pack()
    entry_string2 = tk.Entry(window)
    entry_string2.pack()

    # Process Button
    process_button = tk.Button(window, text="Plot", command=process_inputs)
    process_button.pack()

    # Run the main event loop
    window.mainloop()
