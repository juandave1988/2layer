# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 18:37:30 2024

@author: JUAN RAMIREZ
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution, least_squares
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Global variables to store loaded data
x_data = None
y_data = None
canvas = None

# Function to read data from a TXT file
def read_data(file_path):
    data = np.loadtxt(file_path)
    x = data[:, 0]  # First column
    y = data[:, 1]  # Second column
    return x, y

# Function to calculate resistivity using the Tagg equation
def tagg_model(vect, x):
    k = (vect[1] - vect[0]) / (vect[1] + vect[0])
    h1 = vect[2]
    p1 = vect[0]
    
    y_med = np.zeros_like(x)
    for i, a in enumerate(x):
        sum_val = 0
        for d in range(1, 100):
            p_a = ((k ** d) / np.sqrt(1 + (2 * d * h1 / a)**2)) - ((k ** d) / np.sqrt(4 + (2 * d * h1 / a)**2))
            sum_val += p_a
        y_med[i] = p1 * (1 + 4 * sum_val)
    
    return y_med

# Error function for optimization
def error_function(vect, x, y):
    y_med = tagg_model(vect, x)
    residuals = y - y_med
    return np.sum(residuals**2)  # Sum of squared residuals

# Error function for least squares methods
def least_squares_error(vect, x, y):
    y_med = tagg_model(vect, x)
    return y - y_med  # Residuals for least squares

# Function to apply non-negativity bounds
def apply_bounds(vect):
    vect = np.maximum(vect, 0)  # Ensure values are non-negative
    return vect

# Function to calculate mean squared error
def calculate_mse(x, y, vect):
    y_med = tagg_model(vect, x)
    mse = np.mean((y - y_med) ** 2)
    return mse

# Function to perform optimization with multiple initial points
def optimize(method, lsq_method, x, y, xo_list):
    # Define bounds for p1, p2, and h1
    bounds = [(0, 10000), (0, 10000), (0, 100)]  # Adjust these bounds as necessary

    best_result = None
    best_mse = float('inf')

    for xo in xo_list:
        if method == "Differential Evolution":
            result = differential_evolution(error_function, bounds, args=(x, y))
        elif method == "Nelder-Mead":
            result = minimize(error_function, xo, args=(x, y), method='Nelder-Mead')
            result.x = apply_bounds(result.x)
        elif method == "Powell":
            result = minimize(error_function, xo, args=(x, y), method='Powell')
            result.x = apply_bounds(result.x)
        elif method == "Least Squares":
            result = least_squares(least_squares_error, xo, args=(x, y), method=lsq_method)
            result.x = apply_bounds(result.x)
        
        mse = calculate_mse(x, y, result.x)
        
        if mse < best_mse:
            best_mse = mse
            best_result = result
    
    return best_result

# Function to load data file
def load_data():
    global x_data, y_data
    file_path = filedialog.askopenfilename()
    if file_path:
        x_data, y_data = read_data(file_path)
        entry_file_path.delete(0, tk.END)
        entry_file_path.insert(0, file_path)

# Function to execute optimization and show results
def run_optimization():
    global x_data, y_data, canvas
    method = method_var.get()
    if not method:
        messagebox.showerror("Error", "Select an optimization method")
        return
    
    if x_data is None or y_data is None:
        messagebox.showerror("Error", "Load a valid data file")
        return

    # Initial points (xo)
    xo_list = [
        np.array([p1, p2, h1]) 
        for p1 in np.arange(100, 1100, 100)
        for p2 in np.arange(100, 1100, 100)
        for h1 in np.arange(1, 21)
    ][:10]  # Select the first 10 combinations
    
    # Select specific method for Least Squares
    lsq_method = lsq_method_var.get() if method == "Least Squares" else None

    # Perform optimization
    result = optimize(method, lsq_method, x_data, y_data, xo_list)
    X = result.x

    # Calculate mean squared error
    mse = calculate_mse(x_data, y_data, X)

    # Show results
    results_var.set(f"Model Results:\n"
                    f"Resistivity p1 [ohms.m]: {X[0]:.2f}\n"
                    f"Resistivity p2 [ohms.m]: {X[1]:.2f}\n"
                    f"Depth of the first layer h1 [m]: {X[2]:.2f}\n"
                    f"Mean Squared Error (MSE): {mse:.2f}")

    # Plot results
    y2 = tagg_model(X, x_data)
    fig, ax = plt.subplots()
    ax.plot(x_data, y_data, 'ko', label='Measured Resistivity Curve')
    ax.plot(x_data, y2, 'b-', label='Fitted Model')
    ax.legend()
    ax.set_xlabel('Depth a [m]')
    ax.set_ylabel('Resistivity p [ohm.m]')
    
    # Show plot in the GUI
    if canvas:
        canvas.get_tk_widget().destroy()
    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Function to reset method selection and clear results
def reset():
    global x_data, y_data, canvas
    method_var.set(None)
    lsq_method_var.set(None)
    results_var.set("")
    entry_file_path.delete(0, tk.END)
    x_data, y_data = None, None
    if canvas:
        canvas.get_tk_widget().destroy()

# Create the main window
root = tk.Tk()
root.title("Program for Calculation of Two-Layer Soil Model")

# Create frames
left_frame = tk.Frame(root)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

right_frame = tk.Frame(root)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Title label
label_title = tk.Label(left_frame, text="PROGRAM FOR CALCULATION OF TWO-LAYER SOIL MODEL", font=("Arial", 14, "bold"))
label_title.pack()

# Instructions label
instructions_text = "INSTRUCTIONS:\n\n"\
                    "To begin, press the 'Reset' button. Then, select the data file (.txt) to load.\n"\
                    "Choose an optimization method and execute the optimization."

label_instructions = tk.Label(left_frame, text=instructions_text, justify='left', font=("Arial", 10, "bold"))
label_instructions.pack()

# Interface variables
method_var = tk.StringVar(value=None)  # Initialized to None
results_var = tk.StringVar()
lsq_method_var = tk.StringVar(value=None)  # Initialized to None

# Interface elements
label_method = tk.Label(left_frame, text="Select optimization method:", font=("Arial", 12, "bold"))
label_method.pack()

methods = ["Differential Evolution", "Nelder-Mead", "Powell", "Least Squares"]
for method in methods:
    rb_method = tk.Radiobutton(left_frame, text=method, variable=method_var, value=method)
    rb_method.pack()

# Options for Least Squares
label_lsq_method = tk.Label(left_frame, text="Select Least Squares sub-method:", font=("Arial", 10, "bold"))
label_lsq_method.pack()

lsq_methods = ["trf", "dogbox", "lm"]
for lsq_method in lsq_methods:
    rb_lsq_method = tk.Radiobutton(left_frame, text=lsq_method, variable=lsq_method_var, value=lsq_method)
    rb_lsq_method.pack()

# File path label and entry
label_file_path = tk.Label(left_frame, text="Data file:", font=("Arial", 12, "bold"))
label_file_path.pack()

entry_file_path = tk.Entry(left_frame, width=50)
entry_file_path.pack()

# Load file button
button_load = tk.Button(left_frame, text="Load File", command=load_data)
button_load.pack()

# Run optimization button
button_optimize = tk.Button(left_frame, text="Run Optimization", command=run_optimization)
button_optimize.pack()

# Reset button
button_reset = tk.Button(left_frame, text="Reset", command=reset)
button_reset.pack()

# Results label
label_results = tk.Label(left_frame, textvariable=results_var)
label_results.pack()

# Version and creator label (moved to the bottom)
label_info = tk.Label(left_frame, text="Version 1.0 - Created by Juan David Ramírez - 2024", font=("Arial", 6))
label_info.pack(side=tk.BOTTOM)

# Start the GUI
root.mainloop()
