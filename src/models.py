# Importación de librerias
import pickle
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.models import Sequential

# Lectura de objetos serializados
with open("/home/mapa/Documents/Tec/7S/IA-Avanzada/IA-Avanzada-1/ProyectoFinal/dev/serialObjects/s_regLogModel.pkl", "rb") as file:
    regLogModel:LogisticRegression = pickle.load(file)

with open("/home/mapa/Documents/Tec/7S/IA-Avanzada/IA-Avanzada-1/ProyectoFinal/dev/serialObjects/s_rnModel.pkl", "rb") as file:
    rnmodel:Sequential = pickle.load(file)

with open("/home/mapa/Documents/Tec/7S/IA-Avanzada/IA-Avanzada-1/ProyectoFinal/dev/serialObjects/s_pcaObject.pkl", "rb") as file:
    pcaObject:PCA = pickle.load(file)

with open("/home/mapa/Documents/Tec/7S/IA-Avanzada/IA-Avanzada-1/ProyectoFinal/dev/serialObjects/s_scaler.pkl", "rb") as file:
    scaler:StandardScaler = pickle.load(file)

with open("/home/mapa/Documents/Tec/7S/IA-Avanzada/IA-Avanzada-1/ProyectoFinal/dev/serialObjects/s_categories.pkl", "rb") as file:
    categories:list = pickle.load(file)

# Función que realiza la predicción de una entreada dada
def predict(models, models_id, x_input:np.array, categories):
    # concatenar el input con el vector de zeros y transponerlo
    x_input = x_input.reshape(1,-1)

    # Escalar el vector completo
    x_input_scaled = scaler.transform(x_input)

    # Transformar el vector de entrada proyectandolos a los ejes de los componentes utilizados
    x_input_pca = pcaObject.transform(x_input_scaled)

    # Solo tomar los 10 primeras componentes, tal como entrenó la red
    x_input_pca = x_input_pca[:, :10]

    # Crear figura y subplots
    fig, axs = plt.subplots(len(models),1, figsize=(15, 10))  # 1 fila, 3 columnas

    for i in range(len(models)):
        
        print("======= Predicción Modelo: "+models_id[i])

        if models_id[i] != "RegLog":
            y_pred = models[i].predict(x_input_pca)
            
        else:
            y_pred = models[i].predict_proba(x_input_pca)

        # Convertir a etiquetas predichas tomando la clase con mayor probabilidad
        y_pred_class = np.argmax(y_pred, axis=1)

        # Mostrar resulado
        print("Predicciones por Clase: ", y_pred[0])

        # Mostrar predicción
        print("Categoría Seleccionada: ", str(y_pred_class[0]), " | ", categories[y_pred_class[0]])

        # Plot
        plot = axs[i]
        plot.plot(y_pred[0], color='blue', marker='o')
        plot.set_title("Predicción | "+models_id[i])
        plot.grid(True)
        plot.set_xlabel("Categorías")
        plot.set_ylabel("Probabilidad ("+models_id[i]+")")

        # Reemplazar los números del eje x por las etiquetas
        plot.set_xticks(range(len(categories)))       # posiciones de los ticks
        plot.set_xticklabels(categories, rotation=45) # etiquetas y rotación opcional

    plt.tight_layout()  # Ajusta automáticamente los espacios
    plt.show()

"""

--- VARIABLES ORIGINALES
 0   Destination Port             int64  
 1   Flow Duration                int64  
 2   Total Fwd Packets            int64  
 3   Total Length of Fwd Packets  int64  
 4   Fwd Packet Length Max        int64  
 5   Fwd Packet Length Min        int64  
 6   Fwd Packet Length Mean       float64
 7   Fwd Packet Length Std        float64
 8   Bwd Packet Length Max        int64  
 9   Bwd Packet Length Min        int64  
 10  Bwd Packet Length Mean       float64
 11  Bwd Packet Length Std        float64
 12  Flow Bytes/s                 float64
 13  Flow Packets/s               float64
 14  Flow IAT Mean                float64
 15  Flow IAT Std                 float64
 16  Flow IAT Max                 int64  
 17  Flow IAT Min                 int64  
 18  Fwd IAT Total                int64  
 19  Fwd IAT Mean                 float64
 20  Fwd IAT Std                  float64
 21  Fwd IAT Max                  int64  
 22  Fwd IAT Min                  int64  
 23  Bwd IAT Total                int64  
 24  Bwd IAT Mean                 float64
 25  Bwd IAT Std                  float64
 26  Bwd IAT Max                  int64  
 27  Bwd IAT Min                  int64  
 28  Fwd Header Length            int64  
 29  Bwd Header Length            int64  
 30  Fwd Packets/s                float64
 31  Bwd Packets/s                float64
 32  Min Packet Length            int64  
 33  Max Packet Length            int64  
 34  Packet Length Mean           float64
 35  Packet Length Std            float64
 36  Packet Length Variance       float64
 37  FIN Flag Count               int64  
 38  PSH Flag Count               int64  
 39  ACK Flag Count               int64  
 40  Average Packet Size          float64
 41  Subflow Fwd Bytes            int64  
 42  Init_Win_bytes_forward       int64  
 43  Init_Win_bytes_backward      int64  
 44  act_data_pkt_fwd             int64  
 45  min_seg_size_forward         int64  
 46  Active Mean                  float64
 47  Active Max                   int64  
 48  Active Min                   int64  
 49  Idle Mean                    float64
 50  Idle Max                     int64  
 51  Idle Min                     int64 

"""
