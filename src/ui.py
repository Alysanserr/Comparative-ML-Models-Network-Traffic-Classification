# Importación de Librerías
import tkinter as tk
from tkinter import  filedialog, messagebox
import numpy as np

# Importación de modelos y función de predicción
from models import *


# Variables globales:
models = [ regLogModel, rnmodel ]

models_id = [ "RegLog", "Red Neuronal"]

# Lista de nombres de las variables
feature_names = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Length of Fwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total",
    "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance", "FIN Flag Count", "PSH Flag Count", "ACK Flag Count",
    "Average Packet Size", "Subflow Fwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Max", "Active Min",
    "Idle Mean", "Idle Max", "Idle Min"
]

# ================== UI 

# Función para procesar inputs desde el formulario
def submit_form(entries):
    try:
        x_input = np.array([float(entry.get()) for entry in entries], dtype=float)
        if len(x_input) != 52:
            messagebox.showerror("Error", "Se deben ingresar las 52 variables")
            return
        predict(models, models_id, x_input, categories)
    except ValueError:
        messagebox.showerror("Error", "Ingrese solo valores numéricos")

# Función para procesar archivo .txt
def load_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if not file_path:
        return
    try:
        with open(file_path, 'r') as f:
            data = f.read().strip().split(',')
            if len(data) != 52:
                messagebox.showerror("Error", "El archivo debe contener 52 valores separados por comas")
                return
            x_input = np.array([float(v) for v in data], dtype=float)
            predict(models, models_id, x_input, categories)
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo leer el archivo:\n{e}")

# Crear ventana principal
root = tk.Tk()
root.title("Predicción de Tipo de Actividad en Tráfico de Red")
root.geometry("600x900") 

# Título
title = tk.Label(root, text="Ingrese los valores de las 52 variables:", font=("Arial", 14, "bold"))
title.pack(pady=10)

# Frame para formulario scrolleable
form_frame = tk.Frame(root)
form_frame.pack(fill='both', expand=True, padx=20, pady=10)

canvas = tk.Canvas(form_frame)
scroll_y = tk.Scrollbar(form_frame, orient="vertical", command=canvas.yview)
scroll_frame = tk.Frame(canvas)

scroll_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scroll_frame, anchor='nw')
canvas.configure(yscrollcommand=scroll_y.set)

canvas.pack(side="left", fill="both", expand=True)
scroll_y.pack(side="right", fill="y")

# Crear entradas con labels
entries = []
for i in range(52):
    row_frame = tk.Frame(scroll_frame)
    row_frame.pack(fill='x', pady=2, padx=10)

    label = tk.Label(row_frame, text=f"{feature_names[i]}:", width=35, anchor='w', font=("Arial", 10))
    label.pack(side="left")

    entry = tk.Entry(row_frame, width=20, font=("Arial", 10))
    entry.pack(side="left", padx=5)
    entries.append(entry)

# Botones
btn_frame = tk.Frame(root)
btn_frame.pack(pady=15)

btn_submit = tk.Button(btn_frame, text="Predecir desde formulario", command=lambda: submit_form(entries),
                       bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), padx=10, pady=5)
btn_submit.pack(side="left", padx=10)

btn_load = tk.Button(btn_frame, text="Cargar archivo .txt con 52 variables", command=load_file,
                     bg="#2196F3", fg="white", font=("Arial", 12, "bold"), padx=10, pady=5)
btn_load.pack(side="left", padx=10)

root.mainloop()