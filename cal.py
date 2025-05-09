import tkinter as tk
from tkinter import messagebox
import numpy as np
import sympy as sp
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog

# Custom numerical derivative function
def numerical_derivative(f, x, dx=1e-6, n=1):
    if n == 1:
        return (f(x + dx) - f(x - dx)) / (2 * dx)
    elif n == 2:
        return (f(x + dx) - 2 * f(x) + f(x - dx)) / (dx ** 2)
    else:
        raise ValueError("Only first and second derivatives are supported.")

class CalculusGraphingApp:
    def __init__(self, master):
        self.master = master
        master.title("Calculus Graphing App")
        self.create_widgets()
        self.figure = None
        self.canvas = None

    def create_widgets(self):
        # Function input
        tk.Label(self.master, text="Function (e.g., x**2 + 3*x + 5):").pack()
        self.func_entry = tk.Entry(self.master, width=50)
        self.func_entry.pack()

        # X range
        tk.Label(self.master, text="X start:").pack()
        self.x_start_entry = tk.Entry(self.master)
        self.x_start_entry.pack()

        tk.Label(self.master, text="X end:").pack()
        self.x_end_entry = tk.Entry(self.master)
        self.x_end_entry.pack()

        # Derivative order
        tk.Label(self.master, text="Derivative order:").pack()
        self.order_entry = tk.Entry(self.master)
        self.order_entry.insert(0, "1")
        self.order_entry.pack()

        # Plot button
        self.plot_button = tk.Button(self.master, text="Plot", command=self.plot)
        self.plot_button.pack()

        # Save button
        self.save_button = tk.Button(self.master, text="Save Graph", command=self.save, state=tk.DISABLED)
        self.save_button.pack()

    def plot(self):
        try:
            # Get inputs
            func_str = self.func_entry.get()
            x_start = float(self.x_start_entry.get())
            x_end = float(self.x_end_entry.get())
            order = int(self.order_entry.get())

            # Parse function
            x = sp.symbols('x')
            f_expr = sp.sympify(func_str)
            f = sp.lambdify(x, f_expr, 'numpy')

            # Generate x values
            x_vals = np.linspace(x_start, x_end, 1000)
            y_vals = f(x_vals)

            # Compute derivative
            df = lambda xi: numerical_derivative(f, xi, dx=1e-6, n=order)
            df_vec = np.vectorize(df)
            y_prime = df_vec(x_vals)

            # Compute integral (from x_start to each x)
            integral = np.zeros_like(x_vals)
            for i, xi in enumerate(x_vals):
                integral[i], _ = quad(f, x_start, xi)

            # Plot
            plt.close('all')
            self.figure, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_vals, y_vals, label='Function')
            ax.plot(x_vals, y_prime, label=f'Derivative (order {order})')
            ax.plot(x_vals, integral, label='Integral')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Function, Derivative, and Integral')
            ax.legend()
            ax.grid(True)

            # Embed plot in GUI
            if self.canvas:
                self.canvas.get_tk_widget().destroy()
            self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack()

            self.save_button.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save(self):
        try:
            # Open a file dialog to let user choose where to save
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png")],
                title="Save graph as..."
            )

            # If user cancels the dialog, do nothing
            if not file_path:
                return

            # Save the figure to the selected path
            self.figure.savefig(file_path)
            messagebox.showinfo("Success", f"Graph saved as {file_path}")

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = CalculusGraphingApp(root)
    root.mainloop()