# Importación de librerias
import numpy as np

# Crear muestras aleatoriamente
for i in range(10):
    sample = [
        np.random.randint(1024, 65535),    # Destination Port
        np.random.randint(1000, 1000000),  # Flow Duration
        np.random.randint(1, 1000),        # Total Fwd Packets
        np.random.randint(40, 100000),     # Total Length of Fwd Packets
        np.random.randint(40, 1500),       # Fwd Packet Length Max
        np.random.randint(20, 40),         # Fwd Packet Length Min
        np.random.uniform(30, 1000),       # Fwd Packet Length Mean
        np.random.uniform(0, 500),         # Fwd Packet Length Std
        np.random.randint(40, 1500),       # Bwd Packet Length Max
        np.random.randint(20, 40),         # Bwd Packet Length Min
        np.random.uniform(30, 1000),       # Bwd Packet Length Mean
        np.random.uniform(0, 500),         # Bwd Packet Length Std
        np.random.uniform(0, 1e6),         # Flow Bytes/s
        np.random.uniform(0, 1e4),         # Flow Packets/s
        np.random.uniform(0, 5000),        # Flow IAT Mean
        np.random.uniform(0, 5000),        # Flow IAT Std
        np.random.randint(0, 10000),       # Flow IAT Max
        np.random.randint(0, 1000),        # Flow IAT Min
        np.random.randint(0, 10000),       # Fwd IAT Total
        np.random.uniform(0, 1000),        # Fwd IAT Mean
        np.random.uniform(0, 500),         # Fwd IAT Std
        np.random.randint(0, 1000),        # Fwd IAT Max
        np.random.randint(0, 100),         # Fwd IAT Min
        np.random.randint(0, 10000),       # Bwd IAT Total
        np.random.uniform(0, 1000),        # Bwd IAT Mean
        np.random.uniform(0, 500),         # Bwd IAT Std
        np.random.randint(0, 1000),        # Bwd IAT Max
        np.random.randint(0, 100),         # Bwd IAT Min
        np.random.randint(20, 1500),       # Fwd Header Length
        np.random.randint(20, 1500),       # Bwd Header Length
        np.random.uniform(0, 1e4),         # Fwd Packets/s
        np.random.uniform(0, 1e4),         # Bwd Packets/s
        np.random.randint(20, 1500),       # Min Packet Length
        np.random.randint(20, 1500),       # Max Packet Length
        np.random.uniform(30, 1000),       # Packet Length Mean
        np.random.uniform(0, 500),         # Packet Length Std
        np.random.uniform(0, 500),         # Packet Length Variance
        np.random.randint(0, 10),          # FIN Flag Count
        np.random.randint(0, 10),          # PSH Flag Count
        np.random.randint(0, 100),         # ACK Flag Count
        np.random.uniform(30, 1000),       # Average Packet Size
        np.random.randint(0, 50000),       # Subflow Fwd Bytes
        np.random.randint(0, 100000),      # Init_Win_bytes_forward
        np.random.randint(0, 100000),      # Init_Win_bytes_backward
        np.random.randint(0, 1000),        # act_data_pkt_fwd
        np.random.randint(0, 1000),        # min_seg_size_forward
        np.random.uniform(0, 500),         # Active Mean
        np.random.randint(0, 1000),        # Active Max
        np.random.randint(0, 1000),        # Active Min
        np.random.uniform(0, 500),         # Idle Mean
        np.random.randint(0, 1000),        # Idle Max
        np.random.randint(0, 1000)         # Idle Min
    ]

    # Convertir a string separado por comas
    sample_str = ','.join(map(str, sample))
    file = open(f"../inputSamples/inputSample{i}.txt", "w")

    file.write(sample_str)

    file.close()