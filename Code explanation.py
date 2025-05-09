# Import required modules
import tkinter as tk  # GUI library
from tkinter import messagebox  # For displaying pop-up messages
import numpy as np  # For numerical operations
import sympy as sp  # For symbolic math (like parsing user-input functions)
from scipy.misc import derivative  # For numerical differentiation
from scipy.integrate import quad  # For numerical integration
import matplotlib.pyplot as plt  # For plotting graphs
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Embed matplotlib in tkinter

# Define the main application class
class CalculusGraphingApp:
    def __init__(self, master):
        self.master = master  # Store the main window
        master.title("Calculus Graphing App")  # Set window title
        self.create_widgets()  # Initialize the UI widgets
        self.figure = None  # Placeholder for matplotlib figure
        self.canvas = None  # Placeholder for the figure canvas

# Create the GUI widgets
    def create_widgets(self):
        # Function input label and entry
        tk.Label(self.master, text="Function (e.g., x**2 + 3*x + 5):").pack()
        self.func_entry = tk.Entry(self.master, width=50)
        self.func_entry.pack()

        # X-range inputs
        tk.Label(self.master, text="X start:").pack()
        self.x_start_entry = tk.Entry(self.master)
        self.x_start_entry.pack()

        tk.Label(self.master, text="X end:").pack()
        self.x_end_entry = tk.Entry(self.master)
        self.x_end_entry.pack()

        # Derivative order input
        tk.Label(self.master, text="Derivative order:").pack()
        self.order_entry = tk.Entry(self.master)
        self.order_entry.insert(0, "1")  # Default derivative order is 1
        self.order_entry.pack()

        # Button to plot the graph
        self.plot_button = tk.Button(self.master, text="Plot", command=self.plot)
        self.plot_button.pack()

        # Button to save the graph (initially disabled)
        self.save_button = tk.Button(self.master, text="Save Graph", command=self.save, state=tk.DISABLED)
        self.save_button.pack()

# Function to handle plotting
    def plot(self):
        try:
            # Retrieve user inputs
            func_str = self.func_entry.get()
            x_start = float(self.x_start_entry.get())
            x_end = float(self.x_end_entry.get())
            order = int(self.order_entry.get())

            # Parse the user-defined function string to a symbolic expression
            x = sp.symbols('x')
            f_expr = sp.sympify(func_str)
            f = sp.lambdify(x, f_expr, 'numpy')  # Convert symbolic to numeric function

            # Generate x-values in the given range
            x_vals = np.linspace(x_start, x_end, 1000)
            y_vals = f(x_vals)  # Evaluate the function on x_vals

            # Compute the nth-order derivative using numerical method
            df = lambda xi: derivative(f, xi, dx=1e-6, n=order)
            df_vec = np.vectorize(df)  # Apply the derivative to an array
            y_prime = df_vec(x_vals)  # Derivative values

            # Compute the definite integral from x_start to each x_val
            integral = np.zeros_like(x_vals)
            for i, xi in enumerate(x_vals):
                integral[i], _ = quad(f, x_start, xi)

            # Clear any existing plot
            plt.close('all')
            self.figure, ax = plt.subplots(figsize=(10, 6))

            # Plot the original function, derivative, and integral
            ax.plot(x_vals, y_vals, label='Function')
            ax.plot(x_vals, y_prime, label=f'Derivative (order {order})')
            ax.plot(x_vals, integral, label='Integral')

            # Configure plot labels and title
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Function, Derivative, and Integral')
            ax.legend()
            ax.grid(True)

            # Remove old canvas if it exists
            if self.canvas:
                self.canvas.get_tk_widget().destroy()

            # Embed the new plot into the Tkinter window
            self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack()

            # Enable the save button now that a graph is available
            self.save_button.config(state=tk.NORMAL)

        except Exception as e:
            # Display any error messages to the user
            messagebox.showerror("Error", str(e))

    # Function to save the plot as an image
    def save(self):
        try:
            filename = 'graph.png'  # Default file name
            self.figure.savefig(filename)  # Save the matplotlib figure
            messagebox.showinfo("Success", f"Graph saved as {filename}")  # Notify success
        except Exception as e:
            messagebox.showerror("Error", str(e))  # Show error if saving fails

