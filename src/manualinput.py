# Importación de librerias
import numpy as np

# Importación de modelos y función de predicción
from models import *

# Variables globales:
models = [ regLogModel, rnmodel ]
models_id = [ "RegLog", "Red Neuronal"]

# Crear vector de entrada aleatorio
x_input = np.random.randn(52)  # float con media 0 y desviación estándar 1

# Hacer predicción
predict(models, models_id, x_input, categories)
