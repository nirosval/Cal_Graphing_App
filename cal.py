# ----------------------------------------------------- [CED] -----------------------------------------------------------------------------------
import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
import sympy as sp
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Custom numerical derivative function
def numerical_derivative(f, x, dx=1e-6, n=1):
    if n == 1:
        return (f(x + dx) - f(x - dx)) / (2 * dx)
    elif n == 2:
        return (f(x + dx) - 2 * f(x) + f(x - dx)) / (dx ** 2)
    else:
        raise ValueError("Only first and second derivatives are supported.")


# -------------------------------------------------------- [AARON] --------------------------------------------------------------------------------
class CalculusGraphingApp:
    def __init__(self, master):
        self.master = master
        master.title("📈 Calculus Graphing App")
        master.geometry("850x800")
        master.resizable(True, True)
        self.figure = None
        self.canvas = None

        self.create_widgets()
        # Set default example values
        self.func_entry.insert(0, "x**2")
        self.x_start_entry.insert(0, "-5")
        self.x_end_entry.insert(0, "5")
        self.a_entry.insert(0, "0")
        self.b_entry.insert(0, "2")

    def create_widgets(self):
        # --- Function input frame ---
        input_frame = tk.Frame(self.master, pady=10)
        input_frame.pack()

        tk.Label(input_frame, text="Function:", font=("Arial", 12)).grid(row=0, column=0, sticky="w", padx=5)
        self.func_entry = tk.Entry(input_frame, width=40, font=("Arial", 11))
        self.func_entry.grid(row=0, column=1, padx=5)

        # --- Range frame ---
        range_frame = tk.Frame(self.master, pady=10)
        range_frame.pack()

        tk.Label(range_frame, text="X start:", font=("Arial", 12)).grid(row=0, column=0, padx=5)
        self.x_start_entry = tk.Entry(range_frame, width=10, font=("Arial", 11))
        self.x_start_entry.grid(row=0, column=1, padx=5)

        tk.Label(range_frame, text="X end:", font=("Arial", 12)).grid(row=0, column=2, padx=5)
        self.x_end_entry = tk.Entry(range_frame, width=10, font=("Arial", 11))
        self.x_end_entry.grid(row=0, column=3, padx=5)

        tk.Label(range_frame, text="Deriv Order:", font=("Arial", 12)).grid(row=0, column=4, padx=5)
        self.order_entry = tk.Entry(range_frame, width=5, font=("Arial", 11))
        self.order_entry.insert(0, "1")
        self.order_entry.grid(row=0, column=5, padx=5)

        # --- Definite Integral frame ---
        int_frame = tk.Frame(self.master, pady=5)
        int_frame.pack()

        tk.Label(int_frame, text="Definite Integral:", font=("Arial", 12)).grid(row=0, column=0, padx=5)
        tk.Label(int_frame, text="a:", font=("Arial", 12)).grid(row=0, column=1, padx=2)
        self.a_entry = tk.Entry(int_frame, width=7, font=("Arial", 11))
        self.a_entry.grid(row=0, column=2, padx=2)

        tk.Label(int_frame, text="b:", font=("Arial", 12)).grid(row=0, column=3, padx=2)
        self.b_entry = tk.Entry(int_frame, width=7, font=("Arial", 11))
        self.b_entry.grid(row=0, column=4, padx=2)

        # --- Button frame ---
        button_frame = tk.Frame(self.master, pady=10)
        button_frame.pack()

        self.plot_button = tk.Button(button_frame, text="📊 Plot", command=self.plot, width=12,
                                     font=("Arial", 11, "bold"))
        self.plot_button.grid(row=0, column=0, padx=10)

        self.save_button = tk.Button(button_frame, text="💾 Save Graph", command=self.save, state=tk.DISABLED, width=12,
                                     font=("Arial", 11, "bold"))
        self.save_button.grid(row=0, column=1, padx=10)

        self.help_button = tk.Button(button_frame, text="❓ Help", command=self.show_help, width=12,
                                     font=("Arial", 11, "bold"))
        self.help_button.grid(row=0, column=2, padx=10)

        # --- Results display frame ---
        result_frame = tk.Frame(self.master, pady=10)
        result_frame.pack()

        tk.Label(result_frame, text="📝 Symbolic Results:", font=("Arial", 13, "bold")).pack(anchor="w", padx=10)
        self.result_text = tk.Text(result_frame, height=6, width=100, font=("Courier New", 11), state=tk.DISABLED,
                                   bg="#f8f8f8")
        self.result_text.pack(padx=10, pady=5)

        # --- Plot area frame ---
        self.plot_frame = tk.Frame(self.master, pady=10)
        self.plot_frame.pack()

    def show_help(self):
        """Display user guide"""
        help_text = """📚 HOW TO USE THIS CALCULATOR

1. FUNCTION INPUT:
• Use standard math operations: + - * / 
• Exponents: x**2 (x squared)
• Square roots: sqrt(x)
• Trigonometric functions: sin(x), cos(x), tan(x)
• Logarithms: log(x) (natural), log10(x)
• Constants: pi (3.1415...) and e (2.7182...)

2. EXAMPLE INPUTS:
• Quadratic: x**2 - 4
• Polynomial: 2*x**3 + x**2 - 5
• Trigonometric: sin(2*pi*x)
• Exponential: 2*exp(-x**2)

3. DEFINTE INTEGRAL:
• Enter bounds 'a' and 'b' to calculate area between these points
• Example: a=0 and b=2 for x**2 gives area 8/3

4. TIPS:
• Use parentheses to group operations
• Start with simple functions first!
• Click Plot to update results"""
        messagebox.showinfo("User Guide", help_text)

    # -------------------------------------------------- [NIROS] --------------------------------------------------------------------------------------
    def plot(self):
        try:
            func_str = self.func_entry.get()
            x_start = float(self.x_start_entry.get())
            x_end = float(self.x_end_entry.get())
            order = int(self.order_entry.get())
            a = float(self.a_entry.get())
            b = float(self.b_entry.get())

            x = sp.symbols('x')
            f_expr = sp.sympify(func_str)
            f = sp.lambdify(x, f_expr, 'numpy')

            x_vals = np.linspace(x_start, x_end, 1000)
            y_vals = f(x_vals)

            df = lambda xi: numerical_derivative(f, xi, dx=1e-6, n=order)
            df_vec = np.vectorize(df)
            y_prime = df_vec(x_vals)

            integral = np.zeros_like(x_vals)
            for i, xi in enumerate(x_vals):
                integral[i], _ = quad(f, x_start, xi)

            # Calculate definite integral
            def_integral, def_error = quad(f, a, b)

            plt.close('all')
            self.figure, ax = plt.subplots(figsize=(9, 5.5))
            ax.plot(x_vals, y_vals, label='Function', linewidth=2)
            ax.plot(x_vals, y_prime, label=f'Derivative (order {order})', linestyle='--')
            ax.plot(x_vals, integral, label='Integral', linestyle=':')

            # Shade definite integral area
            x_fill = np.linspace(a, b, 100)
            y_fill = f(x_fill)
            ax.fill_between(x_fill, y_fill, alpha=0.2, label=f'Area ({a} to {b})')

            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('y', fontsize=12)
            ax.set_title('Function, Derivative, and Integral', fontsize=14)
            ax.legend()
            ax.grid(True)

            if self.canvas:
                self.canvas.get_tk_widget().destroy()
            self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack()

            self.save_button.config(state=tk.NORMAL)

            derivative_expr = sp.diff(f_expr, x, order)
            integral_expr = sp.integrate(f_expr, x)
            def_int_expr = sp.Integral(f_expr, (x, a, b))

            result_str = f"f(x) = {sp.simplify(f_expr)}\n"
            result_str += f"{order} derivative: f{("'" * order)}(x) = {sp.simplify(derivative_expr)}\n"
            result_str += f"Indefinite Integral: ∫f(x)dx = {sp.simplify(integral_expr)} + C\n"
            result_str += f"Definite Integral from {a} to {b}: {def_integral:.4f} (Error: ±{def_error:.1e})"

            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, result_str)
            self.result_text.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ----------------------------------------------------- [RONN] -----------------------------------------------------------------------------------
    def save(self):
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png")],
                title="Save graph as..."
            )
            if not file_path:
                return
            self.figure.savefig(file_path)
            messagebox.showinfo("Success", f"Graph saved as {file_path}")

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = CalculusGraphingApp(root)
    root.mainloop()