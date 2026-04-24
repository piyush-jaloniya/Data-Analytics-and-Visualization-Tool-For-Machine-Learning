import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import ttkbootstrap as tb
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from models import train_model, calculate_accuracy
from plots import generate_plot
from utils import load_dataset
from sklearn.model_selection import train_test_split

class MLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Data Analytics & Visualization App")
        
        # Configure main window
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Modern theme
        self.style = tb.Style(theme="pulse")
        self.dataset = None
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.selected_inputs = []
        self.plot_window = None
        
        # Create menu bar
        self.create_menu()
        
        # Main container with grid layout
        self.main_frame = tb.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure grid weights
        self.main_frame.grid_rowconfigure(3, weight=1)  # Data preview expands
        self.main_frame.grid_rowconfigure(5, weight=1)  # Plot controls expand
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # Widgets
        self.create_widgets()

    def create_menu(self):
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Reset", command=self.reset_app)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Help", command=self.show_help)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)

    def create_widgets(self):
        # File Upload Frame (row 0)
        self.frame_upload = tb.LabelFrame(self.main_frame, text="1. Data Upload", padding=10)
        self.frame_upload.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        self.btn_load = tb.Button(
            self.frame_upload, 
            text="Browse Dataset", 
            command=self.load_data,
            bootstyle="primary"
        )
        self.btn_load.pack(side=tk.LEFT, padx=5)
        self.label_file = tb.Label(self.frame_upload, text="No file selected")
        self.label_file.pack(side=tk.LEFT, padx=5)

        # Column Selection Frame (row 1)
        self.frame_columns = tb.LabelFrame(self.main_frame, text="2. Select Columns", padding=10)
        self.frame_columns.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        # Input Features (Listbox for multi-select)
        tb.Label(self.frame_columns, text="Input Features :").pack(anchor=tk.W)
        self.input_listbox = tk.Listbox(
            self.frame_columns, 
            selectmode=tk.MULTIPLE, 
            height=4,
            exportselection=False
        )
        self.input_listbox.pack(fill=tk.X, padx=5)
        
        # Target Column (Combobox for single-select)
        tb.Label(self.frame_columns, text="Target Column:").pack(anchor=tk.W)
        self.output_var = tk.StringVar()
        self.output_dropdown = tb.Combobox(
            self.frame_columns,
            textvariable=self.output_var,
            state="readonly"
        )
        self.output_dropdown.pack(fill=tk.X, padx=5)

        # Model Training Frame (row 2)
        self.frame_model = tb.LabelFrame(self.main_frame, text="3. Model Training", padding=10)
        self.frame_model.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        
        self.model_var = tk.StringVar()
        self.model_dropdown = tb.Combobox(
            self.frame_model,
            textvariable=self.model_var,
            values=["Linear / Multiple Regression", "Logistic Regression", "Decision Tree", "Random Forest", "SVM", "Naive Bayes"],
            state="readonly"
        )
        self.model_dropdown.pack(side=tk.LEFT, padx=5)
        
        self.btn_train = tb.Button(
            self.frame_model,
            text="Train Model",
            command=self.train_model,
            bootstyle="success"
        )
        self.btn_train.pack(side=tk.LEFT, padx=5)
        
        # Training Status and Accuracy
        self.train_status = tb.Label(
            self.frame_model,
            text="🟠 Not Trained",
            bootstyle="warning"
        )
        self.train_status.pack(side=tk.LEFT, padx=5)
        
        self.accuracy_var = tk.StringVar(value="Accuracy: N/A")
        self.label_accuracy = tb.Label(
            self.frame_model,
            textvariable=self.accuracy_var,
            bootstyle="info"
        )
        self.label_accuracy.pack(side=tk.LEFT, padx=5)

        # Data Preview Frame (row 3) - Reduced height
        self.frame_data = tb.LabelFrame(self.main_frame, text="Data Preview", padding=10)
        self.frame_data.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
        
        # Treeview with scrollbars - Limited to 8 rows
        self.tree = ttk.Treeview(self.frame_data, show="headings", height=8)
        self.tree_scroll_y = ttk.Scrollbar(self.frame_data, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree_scroll_x = ttk.Scrollbar(self.frame_data, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=self.tree_scroll_y.set, xscrollcommand=self.tree_scroll_x.set)
        
        self.tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.pack(fill=tk.BOTH, expand=True)

        # Manual Prediction Frame (row 4) - Now with more space
        self.frame_prediction = tb.LabelFrame(self.main_frame, text="4. Manual Prediction", padding=10)
        self.frame_prediction.grid(row=4, column=0, sticky="nsew", padx=5, pady=5)
        
        # Container for input fields (2 columns)
        self.pred_inputs_frame = tb.Frame(self.frame_prediction)
        self.pred_inputs_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create two columns
        self.left_col = tb.Frame(self.pred_inputs_frame)
        self.left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.right_col = tb.Frame(self.pred_inputs_frame)
        self.right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.input_entries = []
        
        # Prediction button and result - Now fully visible
        self.btn_predict = tb.Button(
            self.frame_prediction,
            text="Predict",
            command=self.make_prediction,
            bootstyle="primary",
            state=tk.DISABLED
        )
        self.btn_predict.pack(pady=(10, 5))
        
        self.prediction_result = tb.Label(
            self.frame_prediction,
            text="Prediction: N/A",
            bootstyle="info"
        )
        self.prediction_result.pack(pady=(0, 5))

        # Plot Controls Frame (row 5)
        self.frame_plot_controls = tb.LabelFrame(self.main_frame, text="5. Visualization", padding=10)
        self.frame_plot_controls.grid(row=5, column=0, sticky="ew", padx=5, pady=5)
        
        self.plot_var = tk.StringVar()
        self.plot_dropdown = tb.Combobox(
            self.frame_plot_controls,
            textvariable=self.plot_var,
            values=["Scatter Plot", "Confusion Matrix"],
            state="readonly"
        )
        self.plot_dropdown.pack(side=tk.LEFT, padx=5)
        
        self.btn_plot = tb.Button(
            self.frame_plot_controls,
            text="Generate Plot",
            command=self.generate_plot_window,
            bootstyle="info"
        )
        self.btn_plot.pack(side=tk.LEFT, padx=5)

        # Status Bar
        self.status_var = tk.StringVar()
        self.status_bar = tb.Label(
            self.root, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN,
            padding=5
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx")])
        if file_path:
            try:
                self.dataset = load_dataset(file_path)
                self.label_file.config(text=file_path.split("/")[-1])
                self.update_data_preview()
                
                # Update column selection widgets
                columns = list(self.dataset.columns)
                self.input_listbox.delete(0, tk.END)
                for col in columns:
                    self.input_listbox.insert(tk.END, col)
                self.output_dropdown["values"] = columns
                
                self.status_var.set(f"✅ Loaded: {len(self.dataset)} rows | Columns: {', '.join(columns)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset: {e}")

    def update_data_preview(self):
        self.tree.delete(*self.tree.get_children())
        columns = list(self.dataset.columns)
        self.tree["columns"] = columns
        
        # Configure columns with left alignment
        for col in columns:
            self.tree.heading(col, text=col, anchor=tk.W)
            self.tree.column(col, anchor=tk.W, width=100, stretch=tk.YES)
        
        # Insert data (first 8 rows only)
        for _, row in self.dataset.head(8).iterrows():
            self.tree.insert("", tk.END, values=list(row))

    def train_model(self):
        if self.dataset is None:
            messagebox.showerror("Error", "No dataset loaded!")
            return
        
        # Get selected columns
        self.selected_inputs = [self.input_listbox.get(i) for i in self.input_listbox.curselection()]
        selected_output = self.output_var.get()
        
        if not self.selected_inputs or not selected_output:
            messagebox.showerror("Error", "Select input features AND target column!")
            return
        
        model_name = self.model_var.get()
        if not model_name:
            messagebox.showerror("Error", "Select a model first!")
            return
        
        try:
            X = self.dataset[self.selected_inputs]
            y = self.dataset[selected_output]
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.model = train_model(model_name, self.X_train, self.y_train)
            accuracy = calculate_accuracy(self.model, self.X_test, self.y_test)
            
            # Update UI
            self.accuracy_var.set(f"Accuracy: {accuracy:.2%}")
            self.train_status.config(text="✅ Trained", bootstyle="success")
            self.btn_predict.config(state=tk.NORMAL)
            self.setup_prediction_inputs()
            self.status_var.set(f"Trained {model_name} | Test Accuracy: {accuracy:.2%}")
        except Exception as e:
            self.train_status.config(text="❌ Failed", bootstyle="danger")
            messagebox.showerror("Error", f"Model training failed: {str(e)}")

    def setup_prediction_inputs(self):
        # Clear previous inputs
        for widget in self.left_col.winfo_children():
            widget.destroy()
        for widget in self.right_col.winfo_children():
            widget.destroy()
        self.input_entries = []
        
        # Create input fields in 2 columns
        half = len(self.selected_inputs) // 2
        if len(self.selected_inputs) % 2 != 0:
            half += 1
        
        for i, col in enumerate(self.selected_inputs):
            # Choose left or right column
            col_frame = self.left_col if i < half else self.right_col
            
            frame = tb.Frame(col_frame)
            frame.pack(fill=tk.X, pady=2)
            
            tb.Label(frame, text=f"{col}:", width=15, anchor=tk.W).pack(side=tk.LEFT)
            entry = tb.Entry(frame)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            self.input_entries.append(entry)

    def make_prediction(self):
        try:
            # Get values from input fields
            input_values = []
            for entry, col in zip(self.input_entries, self.selected_inputs):
                val = entry.get()
                if not val:
                    raise ValueError(f"Missing value for {col}")
                input_values.append(float(val))
            
            # Make prediction
            prediction = self.model.predict([input_values])[0]
            self.prediction_result.config(text=f"Prediction: {prediction}")
            self.status_var.set(f"Prediction: {prediction}")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")

    def generate_plot_window(self):
        if self.dataset is None or self.model is None:
            messagebox.showerror("Error", "Load data and train a model first!")
            return
        
        plot_name = self.plot_var.get()
        if not plot_name:
            messagebox.showerror("Error", "Select a plot type first!")
            return
        
        try:
            # Create plot window
            if self.plot_window and self.plot_window.winfo_exists():
                self.plot_window.destroy()
                
            self.plot_window = tb.Toplevel(self.root)
            self.plot_window.title(f"{plot_name} - ML Visualization")
            self.plot_window.geometry("900x700")
            
            # Generate plot
            selected_output = self.output_var.get()
            fig = generate_plot(
                plot_name, 
                self.model, 
                self.dataset, 
                self.selected_inputs, 
                selected_output
            )
            
            # Create canvas
            canvas = FigureCanvasTkAgg(fig, master=self.plot_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add navigation toolbar
            toolbar = NavigationToolbar2Tk(canvas, self.plot_window)
            toolbar.update()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.status_var.set(f"Generated {plot_name} plot in new window")
        except Exception as e:
            messagebox.showerror("Error", f"Plot generation failed: {str(e)}")

    def reset_app(self):
        # Reset all variables and UI
        self.dataset = None
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.selected_inputs = []
        
        # Reset UI elements
        self.label_file.config(text="No file selected")
        self.input_listbox.delete(0, tk.END)
        self.output_var.set("")
        self.model_var.set("")
        self.plot_var.set("")
        self.train_status.config(text="🟠 Not Trained", bootstyle="warning")
        self.accuracy_var.set("Accuracy: N/A")
        self.prediction_result.config(text="Prediction: N/A")
        self.tree.delete(*self.tree.get_children())
        
        # Clear prediction inputs
        for widget in self.left_col.winfo_children():
            widget.destroy()
        for widget in self.right_col.winfo_children():
            widget.destroy()
        self.input_entries = []
        
        # Disable prediction button
        self.btn_predict.config(state=tk.DISABLED)
        
        # Close plot window if open
        if self.plot_window and self.plot_window.winfo_exists():
            self.plot_window.destroy()
        
        self.status_var.set("Application reset")

    def show_help(self):
        help_text = """ML Data Analytics & Visualization App Guide:

1. Load Data: Click 'Browse Dataset' to load a CSV/Excel file
2. Select Columns:
   - Choose input features (multiple selection)
   - Select target column (single selection)
3. Train Model:
   - Select model type from dropdown
   - Click 'Train Model'
4. Make Predictions:
   - Enter values in input fields
   - Click 'Predict'
5. Visualize Results:
   - Select plot type
   - Click 'Generate Plot'

Use the reset button to clear all data."""
        
        messagebox.showinfo("Help", help_text)

    def show_about(self):
        about_text = """ML Data Analytics & Visualization App

Version: 1.0
Developed by: Piyush Jaloniya and Ankush Katoch
Description: A tool for machine learning analysis and visualization

Supports:
- Multiple ML models
- Data visualization
- Interactive predictions"""
        
        messagebox.showinfo("About", about_text)

