#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot
import numpy
import math
import random
import os.path as path
import xlsxwriter
import xlrd
from os import remove
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def generate_bias(length):
    array = []
    for i in range(0, length):
        array.append(round(random.uniform(-0.5, 0.5), 2))
    return array
def generate_range_weights(number_neurons):
    return (1/math.sqrt(number_neurons))

def generate_weights(range_i, range_j, range_weight):
    array = []
    for i in range(0, range_i):
        array.append([])
        for j in range(0, range_j):
            array[i].append(round(random.uniform(-range_weight, range_weight), 2))
    return array

def sigmoide(inputs, weights_matriz, bias_vector):   
    n = []
    nb = []
    sigmoide = []
    for row in weights_matriz:
        value = 0
        for i in range(0, len(row)):
            value += (row[i] * inputs[i])
        n.append(round(value, 2))
    
    for i in range(0, len(n)):
        add_bias = (n[i] + bias_vector[i])
        nb.append(add_bias)
    for value in nb:
        value_sigmoide = round((1/(1+(math.e**-value))), 2)
        sigmoide.append(value_sigmoide)
    return sigmoide

def generate_error_output(getting_values, expected_values):
    error_array = []
    for i in range(0, len(expected_values)):
        result = expected_values[i] - getting_values[i]
        error_array.append(round(result, 2))
    return error_array

def generate_error_hidden(matriz, error_output):
    matriz = numpy.transpose(matriz)
    error_array = []
    for row in matriz:
        value = 0.0
        for i in range(0, len(row)):
            value += row[i] *  error_output[i];
        error_array.append(round(value, 2))
        
    return error_array
    
def sub_array(array):
    new_array = []
    for val in array:
        new_array.append(round((1-val), 2))
    return new_array

def back_propagation(error, sigmoide, output, weiths, L, bias):
    result_array = []
    sub_sigmoide = sub_array(sigmoide)
    output = numpy.transpose(output)
    difference_weights = []
    new_weights = []
    for i in range(0, len(sigmoide)):
        result_array.append(round((-1 * error[i]) * sigmoide[i] * sub_sigmoide[i], 2))
    
    #print(len(result_array), len(output))
    for i in range(0, len(result_array)):
        difference_weights.append([])
        for j in range(0, len(output)):
            difference_weights[i].append(round(result_array[i] * output[j], 2))
    
    for i in range(0, len(difference_weights)):
        new_weights.append([])
        for j in range(0, len(difference_weights[i])):
            new_weights[i].append(round(weiths[i][j]-(L*difference_weights[i][j]), 2))
    # Actualizacion del bias
    for i in range(0, len(bias)):
        value = round(bias[i]-(L * result_array[i]), 2)
        bias[i] = value

    return new_weights


# In[3]:


def keep_values(input_hidden_weights, hidden_output_weights, input_hidden_bias, hidden_output_bias):
    if path.exists("values_new.xlsx"):
        print("Existe")
        remove("values_new.xlsx")
        create_file(input_hidden_weights, hidden_output_weights, input_hidden_bias, hidden_output_bias)
    else:
        print(" No Existe")
        create_file(input_hidden_weights, hidden_output_weights, input_hidden_bias, hidden_output_bias)
       
        
def get_values(input_hidden_weights, hidden_output_weights, input_hidden_bias, hidden_output_bias):
    wb = xlrd.open_workbook("values_new.xlsx")
    ws1 = wb.sheet_by_index(0)
    ws2 = wb.sheet_by_index(1)
    ws3 = wb.sheet_by_index(2)
    ws4 = wb.sheet_by_index(3)
    
    rows_sheet_ws1 = ws1.nrows
    cols_sheet_ws1 = ws1.ncols
    
    rows_sheet_ws2 = ws2.nrows
    cols_sheet_ws2 = ws2.ncols
    
    rows_sheet_ws3 = ws3.nrows
    cols_sheet_ws3 = ws3.ncols
    
    rows_sheet_ws4 = ws4.nrows
    cols_sheet_ws4 = ws4.ncols
    
    for i in range(0, rows_sheet_ws1):
        input_hidden_weights.append([])
        for j in range(0, cols_sheet_ws1):
            input_hidden_weights[i].append(ws1.cell_value(i,j))
            
    for i in range(0, rows_sheet_ws2):
        hidden_output_weights.append([])
        for j in range(0, cols_sheet_ws2):
            hidden_output_weights[i].append(ws2.cell_value(i, j))
    
    for i in range(0, rows_sheet_ws3):
        input_hidden_bias.append(ws3.cell_value(i, 0))
        
    for i in range(0, rows_sheet_ws4):
        hidden_output_bias.append(ws4.cell_value(i, 0))

def create_file(input_hidden_weights, hidden_output_weights, input_hidden_bias, hidden_output_bias):
    wb = xlsxwriter.Workbook('values_new.xlsx')
    ws1 = wb.add_worksheet("input_hidden_matriz")
    ws2 = wb.add_worksheet("hidden_output_matriz")
    ws3 = wb.add_worksheet("input_hidden_bias")
    ws4 = wb.add_worksheet("hidden_output_bias")
    row_input = 0
    row_output = 0
    for row in input_hidden_weights:
        for j in range(0, len(row)):
            ws1.write(row_input, j, input_hidden_weights[row_input][j])
        row_input+= 1

    for row in hidden_output_weights:
        for j in range(0, len(row)):
            ws2.write(row_output, j, hidden_output_weights[row_output][j])
        row_output+= 1

    for i in range(0, len(input_hidden_bias)):
        ws3.write(i, 0, input_hidden_bias[i])

    for i in range(0, len(hidden_output_bias)):  
        ws4.write(i, 0, hidden_output_bias[i])

    wb.close()


# In[4]:


# lectura del archivo csv
data_file = open("mnist_test.csv",'r')
data_list = data_file.readlines()
data_file.close()


# In[5]:


# declaracion de matrices y vectores
input_hidden_matriz = []
hidden_output_matriz = []
inputs_matriz = []
hidden_inputs_array = []
expected_values = []
input_hidden_bias = []
hidden_output_bias = []
hidden_sigmoide = []
output_sigmoide = []
numbers_array = []


length_csv = len(data_list);
number_hidden_neurons = 0
L = 1


# In[ ]:


# lectura de archivo y normalizacion de entradas
for i in range(0, length_csv):
    row_values = data_list[i].split(",")
    numbers_array.append(int(row_values[0]))
    #valor esperado
    image_array = numpy.asfarray(row_values[1:])
    inputs_matriz.append([])
    for j in range(0, len(image_array)):
        if image_array[j] <= 1:
            result = 0.01
        else:
            result = (image_array[j]*0.99)/255
        inputs_matriz[i].append(round(result, 2))

# Obtener el numero de neuronas ocultas        
number_hidden_neurons = round(math.sqrt(len(inputs_matriz[0])*10))

# generacion de matriz de valores esperados
for i in range(0, len(numbers_array)):
    expected_values.append([])
    for j in range(0, 10):
        if(j == numbers_array[i]):
            expected_values[i].append(1)
        else:
            expected_values[i].append(0)
# Verificar si el documento existe, si no existe se generan los numero aleatorios
if path.exists("values_new.xlsx") == False:
    # Generacion de vectores para bias1 y bias2
    input_hidden_bias = generate_bias(number_hidden_neurons)
    hidden_output_bias = generate_bias(10)

    # Generacion de pesos de entrada oculta
    range_weight = generate_range_weights(int(len(inputs_matriz[0])))
    input_hidden_matriz = generate_weights(number_hidden_neurons, len(inputs_matriz[0]), range_weight)
    #print(input_hidden_matriz)
    # Generacion de pesos de oculta salida
    range_weight = generate_range_weights(number_hidden_neurons)
    hidden_output_matriz = generate_weights(10, number_hidden_neurons, range_weight)
    #print(hidden_output_matriz)
    """print(inputs_matriz)
    print(input_hidden_matriz)
    print(hidden_output_matriz)
    """
else:
    print("se llenaran los pesos")
    get_values(input_hidden_matriz, hidden_output_matriz, input_hidden_bias, hidden_output_bias)
    
cont_good = 0
cont_bad = 0
epoch = 0
for i in range(0, len(inputs_matriz)):
    # calculo de sigmoides
    #print(inputs_matriz[i])
    sigmoide_input_hidden = sigmoide(inputs_matriz[i], input_hidden_matriz, input_hidden_bias)
    sigmoide_hidden_output = sigmoide(sigmoide_input_hidden, hidden_output_matriz, hidden_output_bias)
    #print("esta es la salida", sigmoide_hidden_output)
    #print("bias 1 ", input_hidden_bias)
    #print("bias 2 ", hidden_output_bias)
    # comparar el error de salida con el valor esperado
    output_value = max(sigmoide_hidden_output)
    getting_value = sigmoide_hidden_output.index(output_value)
    #print(expected_values[i].index(1), getting_value)
    if expected_values[i].index(1) != getting_value:
        cont_bad += 1
        # calculo de error de salida
        error_output = generate_error_output(sigmoide_hidden_output, expected_values[i])
        error_hidden = generate_error_hidden(hidden_output_matriz, error_output)
        #print(hidden_output_matriz)
        hidden_output_matriz = back_propagation(error_output, sigmoide_hidden_output, sigmoide_input_hidden, hidden_output_matriz, L, hidden_output_bias)
        input_hidden_matriz = back_propagation(error_hidden, sigmoide_input_hidden, inputs_matriz[i], input_hidden_matriz, L, input_hidden_bias)
        #print(len(input_hidden_matriz))
    else:
        cont_good += 1
    epoch += 1
    print("epoch ", epoch)
print("good ", cont_good, "bad ", cont_bad)
    #image_array = numpy.asfarray(inputs_matriz[i]).reshape((28,28))
    #matplotlib.pyplot.imshow(image_array, cmap = 'Greys')

    #print(sigmoide_hidden_output)
    
        
# Generacion de errores de salida


keep_values(input_hidden_matriz, hidden_output_matriz, input_hidden_bias, hidden_output_bias)


# In[ ]:




