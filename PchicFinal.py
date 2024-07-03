# Importación de Librerías.
import tkinter as tk           # Libreria para generar interfaz gráfica.
from tkinter import filedialog # Libreria para generar cuadro de diálogos.
from tkinter import ttk        # Función de la liberia  tkinter para seleccionar cuadros de selección.
import pandas as pd            # Librería para importar y exportar archivos .xlsx y realizar operaciones con matrices 
                               # y dataframes.
import numpy as np             # Librería para realizar operaciones matemáticas y análisis de datos.
from scipy.stats import norm   # Librería para aproximar valores a la distribución normal.
from scipy.cluster import hierarchy # Librería para generar clusters jerárquicos en Python.
import matplotlib.pyplot as plt     # Es una  Librería  especializada en la creación de gráficos dimensionales.
import math                         # Permite el acceso a funciones matemáticas definidas en el estándar C.
from numpy.ma import count
from tkinter import scrolledtext
from tkcalendar import DateEntry

pd.set_option('display.float_format', '{:.7f}'.format)
# Declaración de variables globales que contendrán los valores de similaridad:
# Matrices de nivel de similaridad.
# Vector de nodos significativos.
contenedor_de_la_matriz = []
contenedor_matrix_aa = []
contenedor_lista_matriz = []
contenedor_nodos = []
contenedor_x = []
xx = []
yy = []
contenedor_dxf = []
contenedor_df_concatenado = []
contenedor_valor = []
resultados = []
contendor_diccionario = []
contenedor_f = []
contenedor_sumaI = []
contenedor_rk = []
contenedor__sk = []
contenedor_skk = []
contenedor_rkk = []
fi = []
Ii = []
con_suma = []
contenedor_v = []
contenedor_niveles = []
contenedor_datos_niveles = []
nuevo_vector_final=[]
segundo_maximo= []
maximo=[]
etiquetas_columnas_originales=[]
p= int
global dfn
global cuadro_texto
#  Función para calcular la matriz de nivel 0.
#  Esta función está basada en la ecuación de similitud de Israel Lerman.
#  Propuesta en el documento 'Conceptos fundamentales del Análisis Estadístico Implicativo'
# (ASI) y su soporte computacional CHIC  Página 05.
    

def calcular_similitud(base):
    global dfn
    global column_labels
   
    variables_reales_v4 = []
    contenedor_indice_similaridad = []
    contenedor_matriz = []
    card_values = []
    
    matriz = np.zeros((len(base.columns), len(base.columns)))
    # CREACIÓN DEL CICLO PARA RECORRER TODA LA MATRIZ BINARIA 
    # Y CALCULAR CADA PARÁMETRO DE LA ECUACIÓN DE SIMILARIDAD  DE ISRAEL LERMAN 
    # Propuesto en el documento 'Conceptos fundamentales del Análisis Estadístico Implicativo'
    # (ASI) y su soporte computacional CHIC  Página 05. 
    for i in range(len(base.columns)):
        for j in range(i + 1, len(base.columns)):
            A = base.columns[i]      # X_i es la primera variable que se va a combinar con la variable X_
            B = base.columns[j]      # X_j es la variable X_{i+1} hasta llegar al tamaño de las columnas de la base a usar.
            ai = base[A]             # a_i es la cantidad de unos que existen en la columna X_i.
            aj = base[B]             # a_j es la cantidad de unos que existen en la columna X_j.
                                                   # Verificar si alguna de las columnas tiene todos ceros
            if (ai == 0).all() or (aj == 0).all():
                continue 
            card = np.sum((ai * aj)) # "cadr es la suma de unos que existen dentro de cada columna combinada 'Copresencias'."
            card_values.append(round(card))
            n = len(base)            # n es el número de individuos que constituye la base binaria.
            n_ai = np.sum((ai))      # n_ai es la cantidad de unos que existe en cada columna de la matriz original.
            n_aj = np.sum((aj))      # na_j es la cantidad de unos que existen en la variable X_{i+1}.
            kc = (card - (n_ai * n_aj) / n) / np.sqrt((n_ai * n_aj) / n) 
                                     # El índice Kc es la ecuación que constituye el cálculo.
                                     # De la similitud de Israel Lerman que se muestra en la teoría del ASI pag.5.
            sim = norm.cdf(kc)       # Es la aproximación del índice Kc a la distribución normal con media 0 y varianza 1.
                                     # sim es el valor original de la similitud de Lerman entre un par de variables. 
            contenedor_indice_similaridad.append((A, B, card, kc, sim))
            contenedor_lista_matriz.append((A, B, round(sim, 2)))

            matriz[i, j] = sim
            matriz[j, i] = sim

    contenedor_matriz.append(matriz)
    column_labels = base.columns.tolist()
    
    # Obtener las etiquetas de las columnas y filas restantes
    
    
    dfn = pd.DataFrame(matriz, columns=base.columns, index=base.columns)
    
    dfn.fillna(0, inplace=True)

    # Eliminar columnas con todos ceros
    columnas_no_cero = (dfn != 0).any(axis=0)
    dfn = dfn.loc[:, columnas_no_cero]

    # Eliminar filas con todos ceros
    filas_no_cero = (dfn != 0).any(axis=1)
    dfn = dfn.loc[filas_no_cero, :]
    
    # Crear una nueva matriz compacta sin filas y columnas con todo cero
    nueva_matriz = dfn.values
    
    
    # Actualizar column_labels con las etiquetas de las columnas restantes
    column_labels = dfn.columns.tolist()

    # dfn constituye la matriz de nivel 0, es decir, una matriz simétrica con el valor de las similardiades.
    # de varios individuos con un par de variables .

    
    # dfn constituye la matriz de nivel 0, es decir, una matriz simétrica con el valor de las similardiades.
    # de varios individuos con un par de variables .
    
    # Utlidad de variables 
    
    encabezados = column_labels # Constituye el nombre de las columnas de las variables de una base binaria
    dfn.columns = encabezados   # encabezados de la columna.
    dfn.index = encabezados     # encabezados de la fila.
    segundo_maximo = np.amax(matriz) # contenedor del dataframe valores de similaridad.
    contenedor_de_la_matriz.append(dfn)
    
    
    # data_sim constituye el dataframe con el nombre de las variables y las copresencias estandarizadas.
    # El índice Kc y la similardiad  entre pares de variables dentro de cualquier estudio.
    # Valores de copresencias, copresencias estandarizadas e índices de similaridad. 
    # Este dataframe se muestra el documento Conceptos fundamentales del Análisis Estadístico Implicativo'
    # (ASI) y su soporte computacional CHIC  Página 07.
    
    data_sim = pd.DataFrame(contenedor_indice_similaridad, columns=['var1', 'var2', 'card(ai ∩ aj)', 'kc', 's(ai,aj)'])
    data_sim.fillna(0, inplace=True)
    # dfn constituye la matriz de similiardiad de Nivel 0.
    # esta matriz se obtiene con al ecuacion definida para el calculo de la similaridad de Isarel Lerman.
    # esta matriz se encuenta en el l documento Conceptos fundamentales del Análisis Estadístico Implicativo'
    # (ASI) y su soporte computacional CHIC  Página 08 , Matriz de similaridad al nivel cero .
    resultado_text.delete(1.0, tk.END)
    resultado_text.insert(tk.END, "VALORES DE COPRESENCIAS ESTANDARIZADAS E ÍNDICES DE SIMILARIDAD\n")
    resultado_text.insert(tk.END, str(data_sim) + "\n")
    resultado_text.insert(tk.END, "Matriz de Similaridad: Nivel 0\n ")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
        # Esta opción muestra la matriz completa con division de 7 columnas. 
        resultado_text.insert(tk.END, str(dfn) + "\n")
        # con un una subdivison de 7 variables 
    resultado_text.update_idletasks()
    
   
    # Definición de variables que contendrán los valores necesarios para las matrices de nivel.
    valor_maximo = []
    contenedor_etiquetas1 = []

    segundo_maximo = np.amax(matriz)
    valor_maximo.append(segundo_maximo)

    data_sim = pd.DataFrame(contenedor_indice_similaridad, columns=['var1', 'var2', 'card(ai ∩ aj)', 'kc', 's(ai,aj)'])

    # En este apartado se imprimen las variables que se alojan.
    # Dentro de una sola variable para calcular las submatrices de nivel.
    
    resultado_text.insert(tk.END, f"VARIABLES {column_labels}, VALOR DEL NIVEL: {segundo_maximo}\n")
    resultado_text.insert(tk.END, "--------------------------------------------------------------------------\n")
    

    # INICIO DE MATRIZ CUADRADA
    lista = np.array(contenedor_matriz).flatten().tolist()
    dimension = int(len(lista) ** 0.5)
    contenedor_matrix = []
    # Estos ciclos constituyen la construcción de una nueva matriz con las condiciones 
    # del cálculo de las submatrices de nivel.
    if dimension * dimension != len(lista):
        resultado_text.insert(tk.END, "La lista no tiene una dimensión cuadrada perfecta.\n")
    else:
        matriz_cuadrada = [[lista[i * dimension + j] for j in range(dimension)] for i in range(dimension)]

    for fila in matriz_cuadrada:
        contenedor_matrix.append(fila)

    df = pd.DataFrame(contenedor_matrix)
    
    aa = np.matrix(df)
   

    w = aa.shape[0]
    q = aa.shape[1]

    caracteres_eliminados = []
 # Se construye la matriz de nivel 1 con la condición de seleccionar las variables con mayor similitud,
 # eliminar de la matriz 0 y generar una nueva matriz para seleccionar los máximos entre combinaciones
 # de las variables seleccionadas y elevar a 2 por el nivel mencionado en el documento Conceptos fundamentales del Análisis Estadístico Implicativo'
 # (ASI) y su soporte computacional CHIC  Página 08.
 # Una vez obtenidos estos índices de similaridad al nivel cero, las agrupaciones
 # se hacen como en la clasificación clásica:
 # Se construye una primera matriz con los Índices de Similaridad ( )ji s , a,a
 #obtenidos a partir de las combinaciones de todas las variables y según la
 #fórmula (1) o (1a).
 #Se buscan los nuevos Índices de Similaridad al combinar la clase ( )ji a,a , con
 #mayor índice en el paso anterior, con (Lerman, 1970):

         
           


    for i in range(w):
        for j in range(q):
            indice_max = np.unravel_index(np.argmax(aa), aa.shape)
            fila_eliminada = np.ravel(aa[0:, indice_max[0]])
            nuevo_encabezado = np.delete(column_labels, indice_max, axis=0)
            vector1 = np.array(column_labels)
            vector2 = np.array(nuevo_encabezado)

            nuevo_vector = np.setdiff1d(vector1, vector2)
            resultado_vector = ','.join(nuevo_vector)
            variables_reales_v4 = [column_labels[i] for i in range(len(column_labels)) if column_labels[i] in nuevo_vector]
            vector3 = f"F{i} ({', '.join(variables_reales_v4)})"
            nuevo_vector_final = [vector3] + nuevo_encabezado.tolist() 

    resultado_text.insert(tk.END, f"VARIABLES {nuevo_vector_final},  VALOR DEL NIVEL: {segundo_maximo}\n")
    resultado_text.insert(tk.END, "--------------------------------------------------------------------------------j\n")
    
    # Impresión de las variables alojadas en la nueva variable con la matriz correspondiente al nivel.
    vector3 = ""  
    variables_reales_v4 = [column_labels[i] for i in range(len(column_labels)) if column_labels[i] in nuevo_vector] 
   # Se muestran las variables alojadas en una sola variable de agrupación.
    resultado_text.insert(tk.END, f"Las variables reales contenidas en {vector3} son: {', '.join(variables_reales_v4)}\n")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  
        # Esta opción muestra la matriz completa.
        resultado_text.insert(tk.END, str(vector3) + "\n\n")
    resultado_text.update_idletasks()
 
    
   
     
    contenedor_datos_niveles.append((0, segundo_maximo, nuevo_vector_final, vector3, None))


    

  # Construcción de matrices de nivel con sus respectivas etiquetas.
  # Cada vez que se encuentre una matriz de similitud en el nivel, se realiza el
  # algoritmo de agrupamiento de Israel Lerman.
  
    colum_eliminada = aa[:, indice_max[1]]
    columna_eliminada = np.transpose(colum_eliminada)
    fila_eliminada = aa[indice_max[0], :]

    indice_maxa = np.argmax(columna_eliminada)
    indice_mina = np.argmin(columna_eliminada)
    indice_maxb = np.argmax(fila_eliminada)
    indice_minb = np.argmin(fila_eliminada)

    for k in range(len(colum_eliminada)):
        vector1 = np.delete(colum_eliminada, [indice_maxa, indice_mina])
        vector2 = np.delete(fila_eliminada, [indice_maxb, indice_minb])
        vector3 = np.delete(colum_eliminada, [indice_maxa, indice_mina])
        maximos_por_posicion = np.maximum.reduce([vector1, vector2, vector3])

    p = (2 * 1) # p constituye el valor del nivel al que se eleva el máximo de las combinaciones que existen en 
                # la nueva variable alojada con mayor similardiad.
    nuevo_nn = np.array(maximos_por_posicion ** (p)) 
                # En esta línea de código se eleva la variable que contiene las combinaciones de los valores.
    matrix_uni = np.delete(np.delete(aa, indice_max, axis=0), indice_max, axis=1) 
    matriz_n = np.zeros((len(matrix_uni) + 1, len(matrix_uni) + 1))
    matriz_n[1:, 1:] = matrix_uni
    matriz_n[0, 1:] = nuevo_nn
    matriz_n[1:, 0] = nuevo_nn
    matriz_uni = matriz_n
    nueva_matriz = matriz_uni
    matriz_simetrica = np.array(nueva_matriz)
    maximo = np.amax(matriz_simetrica)
     
    # Obtener la variable anterior y la nueva variable generada
    variable_anterior = nuevo_vector_final[0]
    nueva_variable = nuevo_vector_final[-1]
     
    # Crear una lista de etiquetas colocando primero la variable anterior
    nombres = [variable_anterior] + [nombre for nombre in nuevo_vector_final[1:-1]] + [nueva_variable]

    contenedor_etiquetas1.append(nombres)

    # Crear el DataFrame con las etiquetas ordenadas
    dfss = pd.DataFrame(matriz_simetrica, index=nombres, columns=nombres)
   # En este procedimiento se realiza el algoritmo y se construye la matriz con las etiquetas combinadas, es decir, calcular
   # de forma numérica la matriz de nivel k y agregar etiquetas la cual se imprime en el documento 
   # Conceptos fundamentales del Análisis Estadístico Implicativo'(ASI) y su soporte computacional CHIC  Página 08.
    w1 = matriz_simetrica.shape[0]
    q1 = matriz_simetrica.shape[1]

    for i in range(w1):
        for j in range(q1):
            indice_max = np.unravel_index(np.argmax(matriz_simetrica), matriz_simetrica.shape)
            fila_eliminada = np.ravel(matriz_simetrica[0:, indice_max[0]])
            nuevo_encabezado = np.delete(nombres, indice_max, axis=0)
            vector1 = np.array(nombres)
            vector2 = np.array(nuevo_encabezado)
            variables_contenidas = [var for var in nuevo_encabezado if not var.startswith(("A", "B", "C", "D"))]
            nuevo_vector = np.setdiff1d(vector1, vector2)
            resultado_vector = ','.join(nuevo_vector)
            variables_reales_v5 = [column_labels[i] for i in range(len(column_labels)) if column_labels[i] in nuevo_vector]
            vector3 = f"A{i}({', '.join(variables_reales_v5)})"
            nuevo_vector_final = [vector3] + nuevo_encabezado.tolist() 
            
            
            
           
   
            
    resultado_text.insert(tk.END, "MATRIZ NIVEL 1\n")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  
        # Esta opción muestra la matriz completa
        resultado_text.insert(tk.END, str(dfss) + "\n")
    resultado_text.insert(tk.END, "----------------------------------------------------------\n")
    resultado_text.insert(tk.END, f"VARIABLES {nuevo_vector_final} = {vector3}\n")
    resultado_text.insert(tk.END, f"VARIABLES {nuevo_vector_final}, VALOR DEL NIVEL : {maximo}\n")
    variables_reales_v5 = [column_labels[i] for i in range(len(column_labels)) if column_labels[i] in nuevo_vector]
    
    resultado_text.insert(tk.END, f"Las variables reales contenidas en {vector3} son: {', '.join(variables_reales_v5)}\n")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
        # Esta opción muestra la matriz completa
        resultado_text.insert(tk.END, str(vector3) + "\n")
    resultado_text.update_idletasks()
    # dfss constituye la nueva matriz completa con todas las características que conforman las matrices de nivel.
    # Estas matricez se observan en el documento
    # Conceptos fundamentales del Análisis Estadístico Implicativo'(ASI) y su soporte computacional CHIC  Página 08.
    # Indican una reprentacion numerica del Arbol de similaridada de Isarel Lerman.

    etiquetas = nuevo_vector_final
    etiquetasq = etiquetas.copy()

    contenedor_var = []
    longitud = len(matriz_simetrica)
    contenedor0 = []
    # Para niveles superiores a 1, se realiza un bucle que realice el algoritmo de agrupación numérica indicado 
    # en los documentos de (CHIC) y (ASI).
    # . El algoritmo se repite hasta que se utilicen todas las variables 
    # y no queden variables con una similitud máxima.
    # Se construye la matriz de nivel 1 con la condición de seleccionar las variables con mayor similitud,
    # eliminar de la matriz 0 y generar una nueva matriz para seleccionar los máximos entre combinaciones
    # de las variables seleccionadas y elevar a 2 por el nivel mencionado en el documento Conceptos fundamentales del Análisis Estadístico Implicativo'
    # (ASI) y su soporte computacional CHIC  Página 08.
    
    
    for i in range(longitud):
        l =0  
       
        k= 1   # Define el número de nivel en la matriz (etiqueta).
        
        while (len(matriz_uni) > 1): # Si las variables (longitud) son mayores a hacer matriz uni mayor a niveles.
            l =  l+1
            k = k+1                  # "Recorrerá k + 1 hasta ocupar todas las variables.
            
                                     #  "Verificar si la matriz es toda cero.
            if np.all(matriz_uni == 0):
                break

            indice_max = np.unravel_index(np.argmax(matriz_uni), matriz_uni.shape)
            colum_eliminada = matriz_uni[:, indice_max[1]]
            columna_eliminada = np.transpose(colum_eliminada)
            fila_eliminada = matriz_uni[indice_max[0], :]

            indice_maxa = np.argmax(columna_eliminada)
            indice_mina = np.argmin(columna_eliminada)
            indice_maxb = np.argmax(fila_eliminada)
            indice_minb = np.argmin(fila_eliminada)

            vector1 = np.delete(columna_eliminada, [indice_maxa, indice_mina])
            vector2 = np.delete(fila_eliminada, [indice_maxb, indice_minb])
            vector3 = np.delete(columna_eliminada, [indice_maxa, indice_mina])
            maximos_por_posicion = np.maximum.reduce([vector1, vector2, vector3])
            # Procedimiento realizado para el cálculo de niveles superiores especificados
            # en pasos anteriores.
            p = (2 * l) # P es el número del nivel por el cual el máximo de las combinaciones será elevado,
                        # es decir, (HIP , PUNK ) * (JAZ ) = MAX(HIP ,JAZ , PUNK , JAZ ^ P).
            nuevo_nn = np.array(maximos_por_posicion ** (p))
            matriz_uni = np.delete(np.delete(matriz_uni, indice_max, axis=0), indice_max, axis=1)
            matriz_n = np.zeros((len(matriz_uni) + 1, len(matriz_uni) + 1))
            matriz_n[1:, 1:] = matriz_uni
            matriz_n[0, 1:] = nuevo_nn
            matriz_n[1:, 0] = nuevo_nn
            # generacion de nuevas matricez 
            
            if matriz_uni.shape[0] > 2 and matriz_uni.shape[1] > 2:
               matriz_sin_maximos = matriz_n[1:, 1:]  # Tomar la submatriz sin los valores máximos.
               maximo = np.amax(matriz_sin_maximos)   # Calcular el máximo valor de similitud de la submatriz.
            else:
               maximo = np.amax(matriz_n)  

            
            
            
            matriz_uni = matriz_n

            nueva_matriz = matriz_uni
            matriz_simetrica = np.array(nueva_matriz)
            maximo = np.amax(matriz_simetrica)

            nombres = [f'{etiquetasq[i]}' for i in range(matriz_simetrica.shape[0])]

            dfss = pd.DataFrame(matriz_simetrica, index=nombres, columns=nombres)

            w2 = matriz_simetrica.shape[0]
            q2 = matriz_simetrica.shape[1]

            for i in range(w2):
                for j in range(q2):
                    indice_max = np.unravel_index(np.argmax(matriz_simetrica), matriz_simetrica.shape)
                    fila_eliminada = np.ravel(matriz_simetrica[0:, indice_max[0]])
                    nuevo_encabezado = np.delete(nombres, indice_max, axis=0)

                    vector1 = np.array(nombres)
                    vector2 = np.array(nuevo_encabezado)

                    nuevo_vector = np.setdiff1d(vector1, vector2)
                    resultado_vector = ','.join(nuevo_vector)
                    vector3 = ""
                    variables_contenidas= " "
                    variables_reales_v6 = [column_labels[i] for i in range(len(column_labels)) if column_labels[i] in nuevo_vector]
                    vector3=f"M{l} ({', '.join(variables_reales_v6)}) "
                    nuevo_vector_final = nuevo_encabezado.tolist()
                    nuevo_vector_final.insert(0, vector3)
                   
             # Agregación de etiquetas y creación de matrices desde el nivel 1 hasta el nivel K-1.
             # NOTA :
             # Estas funciones constituyen la elaboración de filas, columnas, y etiquetas de las matrices de 
             # nivel de similitud, las cuales conforman una matriz de nivel 0 acompañada del cálculo numérico.
                
            etiquetas = nombres
            etiquetasq = [etiqueta for etiqueta in etiquetas if etiqueta not in (nuevo_vector)]
            etiquetasq.insert(0, vector3)        
                    
          
            
            
             
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # Esta opción muestra la matriz completa
               
               datos_nivel = (k, maximo, nuevo_vector, vector3, dfss)
               contenedor_datos_niveles.append(datos_nivel)
               
               
            resultado_text.update_idletasks()
              
    

            resultado_text.see(tk.END)  
           # Desplazar hacia abajo para mostrar los resultados más recientes.
           # Verificar si la matriz es completamente cero después de las operaciones.
            if np.all(matriz_uni == 0):
                break
           # Fin del algoritmo de submatrices de nivel.
           # Especificado en el documento documento Conceptos fundamentales del Análisis Estadístico Implicativo'
           # (ASI) y su soporte computacional CHIC  Página 08.
           # Se construye una primera matriz con los Índices de Similaridad ( )ji s , a,a
           # obtenidos a partir de las combinaciones de todas las variables y según la
           # fórmula (1) o (1a).
           #S e buscan los nuevos Índices de Similaridad al combinar la clase ( )ji a,a , con
           #m ayor índice en el paso anterior, con (Lerman, 1970):

   
### Creación del dendrograma  
### El dendrograma es la representación gráfica 
### del análisis numérico de submatrices de nivel. La idea es representar los grupos alojados
### en las variables alternas generadas en el cálculo de las matrices de nivel como se representa en el documento 
###  Conceptos fundamentales del Análisis Estadístico Implicativo' pagina 9. 
    def dendograma_invertido(m):
        nueva_matriz1 = np.fill_diagonal(m, 1) 
        # DENDOGRAMA INVERTIDO
        similaridad = hierarchy.distance.pdist(matriz)
        enlaces = hierarchy.linkage(similaridad, method='complete', metric='euclidean')
        # Usa métodos propios de Python para generar dendrogramas,
        dendrogram = hierarchy.dendrogram(enlaces, labels=column_labels, orientation='bottom')
        # Muestra los valores entre las parejas de variables.
        for i, d, c in zip(dendrogram['icoord'], dendrogram['dcoord'], dendrogram['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            plt.plot(x, y)
            
        #  Configura los ejes y muestra el gráfico.
        #   funciones para :
        
        plt.title('Dendrograma')          # Agrega el nombre al gráfico.
        plt.xlabel('Índices de muestra')  # Agrega el nombre al gráfico en el eje x.
        plt.ylabel('SIMILARIDAD')         # Agrega el nombre al gráfico. en el eje Y
        plt.savefig('Dendo.pdf')          # Guarda el dendrograma en un archivo PDF para posteriores visualizaciones y análisis.
        plt.show()                        # Imprime el dedrograma en la pantalla principal.
        data_sim.to_excel('values.xlsx', index=True)  
                                          # Guarda el dataframe original que contiene el cálculo de similitud según Israel Lerman.

        dfn.to_excel('similaridad.xlsx', index=True , header=True) 
                                          # Guarda la matriz de nivel 0 en un archivo Excel.

    dendograma_invertido(matriz)

 
    
# Funcion para la reación de la matriz de nodos significativos.
# calculo disponible en el documento Conceptos fundamentales del Análisis Estadístico Implicativo'
# (ASI) y su soporte computacional CHIC  Página 10 .
#  CHIC determina los nodos significativos para la
#  clasificación obtenida, los cuales se muestran gráficamente con trazos gruesos y
#  rojos (figura 1). Para el ejemplo desarrollado, los nodos significativos se dan al
#  nivel k +1 . Veamos cómo se determinaron estos nodos.

#### Programacion Calculo nodos significativos ######

## debido ala complejidad de formulas se crea una secuencia de pasos acompañado 
## de ecucaciones definidas en el documento de ASI Y CHIC , pag 10
## Indice centrado Como en el análisis de las semejanzas de Lerman (Lerman, 1970) este índice
## sirve de "estadística global de los niveles". Sus variaciones son consideradas para
## significar la constitución de un nivel significativo.

### Pasos ###

# 1) Ordenar por grupos las variables con mayor similitud de forma ascendente.
# 2) Agregar índices a cada par de variables que se unen por similitud.
# 3) Desglosar los parámetros definidos en la ecuación del índice centrado 
#    definido en el documento de ASI página 10. Con este procedimiento se intenta replicar 
#    la forma del cálculo de nodos significativos especificados en las páginas 12 y 13 del mismo documento.
# 4) Mediante el desglosamiento de los parámetros de la ecuación y una vez obtenidas todas las variables,
#    se constituye la matriz de nodos significativos.
# 5) Encontrar el indice Se llama nodo significativo cualquier nodo formado a un nivel que 
#    corresponde a un máximo local de v(omega.k), = s(omega,k )− s(omega,k-1) ecuasion disponible en la 
#    pagina 11 defincion 2 nodos significativos . 
# 6) crear la matriz de nodos significativos y graficar V(omega,k)

def ordenar_nodos(datos):
    # Limpiar contenedores
    contenedor_valor.clear()
    resultados.clear()
    contendor_diccionario.clear()
    contenedor_f.clear()
    contenedor_sumaI.clear()
    contenedor_rkk.clear()
    contenedor_skk.clear()
    fi.clear()
    Ii.clear()
    contenedor_v.clear()
   
    #  Agregar en una variable el dataframe 'data_sim' para ordenar y crear grupos por similaridad.

    dfnuevo = pd.DataFrame(datos)         # Convertir los datos en un DataFrame.
    v_1 = (dfnuevo[0] + "," + dfnuevo[1]) # Unir las columnas de variables encabezad.
    v_2 = (dfnuevo[1] + "," + dfnuevo[0]) # Unir la variables de las columna valor.
    a = np.array(v_1)                     # Transformacion en un array.
    b = np.array(v_2)                     # Transformacion en un array. 
    c = np.array(dfnuevo[2])              # Transformacion en un array.
                                          # Obtener el orden de 'c' de mayor a menor.
    orden = np.argsort(c)[::-1]
    # Actualizar las filas 'a' y 'b' en función del orden de 'c'.
    a_ordenada = a[orden]
    b_ordenada = b[orden]
    c_ordenada = c[orden]
    # Crear un nuevo DataFrame con las filas ordenadas.
    df_ordenados = pd.DataFrame({'Var1': a_ordenada, 'Var2': b_ordenada, 'Valor': c_ordenada})
    df_ordenadoc = pd.DataFrame({'Var1': a_ordenada, 'Var2': b_ordenada})
    df_ordenadov = pd.DataFrame({'Valor1': c_ordenada, 'Valor2': c_ordenada})

    df_combined1 = pd.DataFrame(df_ordenadoc.values.reshape(-1), columns=['Variables'])
    df_combined2 = pd.DataFrame(df_ordenadov.values.reshape(-1), columns=['Valor'])
    df_concatenado = pd.concat([df_combined1, df_combined2], axis=1)

    contenedor_valor.append(df_combined2)

    c_ordenado = df_concatenado['Valor'].values 

    c_orden = pd.Series(c_ordenado)
    # Creacion de grupos basados en los valores únicos.
    grupos = c_orden.groupby(c_orden).groups
    num = 0
   
    
    # En este proceso se ordenan y combinan las variables con mayor similitud.
    # En este punto se desglosan las variables, es decir, (HiP PUNK)(PUNK HIP).
    for i, grupo in enumerate(grupos, 1):
        yy.append(grupo)
    xx.append(grupos)
    # Agregar numeración a los grupos
    diccionario = xx[0]
    contenedor_valores = []
    contendor_diccionario.append(diccionario)

    dados = contendor_diccionario[0]
    # Imprimir los grupos
    w = 0
    for chave, valor in dados.items():
        w = w + 1
        lista = valor
        indice = pd.Index(valor)
        valores = indice.values.tolist()
        contenedor_valores.append(valores)
    # Agregar índices a cada combinación de variables para aplicar el desglosamiento de los parámetros que contiene 
    # la ecuación
    # de la cardinalidad de (Sk) y Rk, el índice centrado S(omega,k)).
    variables = contenedor_valores

    def tabla_resolucion_nodos(variables, contenedor):
        # Utilidad de cada paramtero : 
        v_alpha_k = 0
        s_k = len(contenedor) # Toma el valor del tamaño de los índices agregados
        suma_total = 0
        mk = len(variables)   # Define el tamaño de las variables.
        for i in range(mk):
            contar = count(variables[i])
            resultados.append(contar)
            suma_total += contar
            f = (count(variables[i]) - 1) # f Define el tamaño de las variables - 1 
            contenedor_sumaI.append(suma_total)
            contenedor_f.append(f)
        resultado_text.delete(1.0, tk.END)  # Limpiar el texto antes de agregar nuevos resultados
        resultado_text.insert(tk.END, " vector de nodos significativos\n")
        fila_invertida = contenedor_sumaI[::-1]
        columna_invertida = contenedor_f[::-1]
        s_k = (len((contenedor)) - 1)
        con_al = []
        contt=0
        
        for i in range(len(contenedor_f)):
            fi.append(columna_invertida[i])
            Ii.append(fila_invertida[i])
           
            rk = i + 1   # Es el complemento que existe al seleccionar un índice de cada grupo con el mayor índice,
                         # es decir, si el índice mayor es 20 en algún grupo, se selecciona un índice con Sk=10. Rk 
                         # será igual a 10.
            sk = s_k - i # Es indice mayor que existe en cada grupo .
            
            contenedor_rkk.append(rk)
            contenedor_skk.append(sk)
            card = (sum(Ii) - rk * ((rk + 1) / 2) - sum(fi)) 
            # Se calcula con la siguiente fórmula: Card[G(Ω) ∩ [S∏k * R∏k] = ∑rk_j=1 ij − rk∗(rk+1)2 − ∑rkj=1 fj
            # esto se logra sumando todos los elementos que se encuentran ala izquierda de cada
            # grupo y que nos se encuentran en el conjunto  esta ecuacion se encuentra definida en el documento 
            # Conceptos fundamentales del Análisis Estadístico Implicativo'(ASI) y su soporte computacional CHIC 
            # (ASI) y su soporte computacional CHIC  Página 12.

            s_beta_k = round((card - (0.5 * sk * rk)) / math.sqrt((sk * rk * (sk + rk + 1)) / 12), 5) 
            # S(Ω, k) = [G(Ω)∩[S∏k ∗R∏k]]−1/2
            # SkRk(sk+rk+1)/12 sirve para crear el data frame con V(Ω, k) para posteriormente graficar.
            con_al.append(s_beta_k) 
            # Ecuasiones obtendiad documento 
            # Conceptos fundamentales del Análisis Estadístico Implicativo'(ASI) y su soporte computacional CHIC 
            # (ASI) y su soporte computacional CHIC  Página 11 y 12 definicion 1 y 2.
            valoresv = con_al
            resultados_v = []
            contenedor_vector = []
            for i in range(1, len(valoresv)):
                resta = valoresv[i] - valoresv[i - 1] # S(Ω,K)−S(OMEGA,K −1).
                contenedor_vector.append(resta)
            contenedor_vector.insert(0, valoresv[0])
            for j in range(len(contenedor_vector)):
                v = round(contenedor_vector[j], 3)
            contenedor_v.append(v)
            resultado_text.insert(tk.END, f"Card() [{i}] : {card}, S(Ω,k) [{i}]: {s_beta_k}, V(Ω,k)[{i}]: {v}\n")
        resultado_text.update_idletasks()

    tabla_resolucion_nodos(variables, contenedor_valor[0])

# Grafico del vector V(omega,k) nodos significativos 
def grafica_omega(v):
    n = len(v)
    x = list(range(0, n))
    y = v
    nuevos_y = [valor if valor >= 0.25 else 0 for valor in v]  # 
    colores = ['red' if valor > 0.25 else 'black' for valor in v]  
    # Lista de colores

    plt.plot(x, y, 'o-', color='black', alpha=0.5)  
    # Línea que une los puntos (negro opaco)

    for i in range(len(x)):
        if colores[i] == 'black':
            plt.scatter(x[i], y[i], c='black', label='Valores de V(Ω,k)', alpha=0.5)  
            # Puntos negros
            plt.text(x[i], y[i], str(y[i]), ha='center', va='bottom', color='black') 
            # Etiqueta para valores de V(Ω,k)
        else:
            plt.scatter(x[i], nuevos_y[i], c='red', label='Nodos significativos', alpha=0.5)  
            # Puntos rojos
            plt.text(x[i], nuevos_y[i]+0.02, str(y[i]), ha='center', va='bottom', color='red') 
            # Etiqueta para nodos significativos

    plt.axhline(y=0.25, color='black', linestyle='--')
    plt.xlabel('Índice de k')
    plt.ylabel('Valor de V(Ω,k)')
    plt.title('Gráfico de V(Ω,k)')
    plt.tight_layout()
    plt.savefig('nodos.pdf')
    plt.show()



def seleccionar_archivo():
    ventana_seleccionar = tk.Toplevel(ventana)
    ventana_seleccionar.title("Seleccionar archivo CSV")
    ventana_seleccionar.geometry("400x150")  # Tamaño de la ventana

    frame = ttk.Frame(ventana_seleccionar)
    frame.pack(padx=10, pady=10)

    label_archivo = ttk.Label(frame, text="Seleccionar archivo CSV:")
    label_archivo.grid(row=0, column=0, padx=5, pady=5)

    entry_archivo = ttk.Entry(frame, width=30)
    entry_archivo.grid(row=0, column=1, padx=5, pady=5)

    boton_buscar = ttk.Button(frame, text="Buscar", command=lambda: buscar_archivo(entry_archivo))
    boton_buscar.grid(row=0, column=2, padx=5, pady=5)

    def buscar_archivo(entry):
        archivo_csv = filedialog.askopenfilename(filetypes=[("Archivos CSV", "*.csv")])
        if archivo_csv:
            entry.delete(0, tk.END)
            entry.insert(0, archivo_csv)
            procesar_archivo(archivo_csv)

    def procesar_archivo(archivo_csv):
        try:
            global base
            base = pd.read_csv(archivo_csv, sep=';')
            base = base.loc[:, (base != 0).any(axis=0)]
            resultado_text.delete(1.0, tk.END)
            mostrar_variables()
            ventana_seleccionar.destroy()
        except Exception as e:
            resultado_text.delete(1.0, tk.END)
            resultado_text.insert(tk.END, f"Error al procesar: {str(e)}")
            
def salir():
    ventana.quit() 

        
# Función para mostrar variables y seleccionarlas para calcular la similitud.

def mostrar_variables():
    if len(base.columns) > 10:
        ventana_variables = tk.Toplevel()
        ventana_variables.title("Seleccionar Variables")

        frame = ttk.Frame(ventana_variables)
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        canvas = tk.Canvas(frame, height=200)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        var_selection = []

        for i, variable in enumerate(base.columns):
            # Filtrar columnas sin nombre (como "Unnamed")
            if variable.startswith("Unnamed"):
                continue
            
            var_var = tk.BooleanVar()
            var_checkbutton = tk.Checkbutton(scrollable_frame, text=variable, variable=var_var)
            var_checkbutton.grid(row=i, column=0, sticky="w")
            var_selection.append((variable, var_var))

        # Botón para seleccionar todas las variables
        var_select_all_var = tk.BooleanVar()
        checkbutton_select_all = tk.Checkbutton(ventana_variables, text="Seleccionar Todas", variable=var_select_all_var, command=lambda: select_all_variables(var_selection, var_select_all_var))
        checkbutton_select_all.grid(row=i+1, column=0, sticky="w")

        boton_calcular = tk.Button(ventana_variables, text="Calcular Similaridad", command=lambda: calcular_similaridad_desde_seleccion(var_selection))
        boton_calcular.grid(row=i+2, column=0, pady=10)
    else:
        ventana_variables = tk.Toplevel()
        ventana_variables.title("Seleccionar Variables")

        var_selection = []

        for variable in base.columns:
            # Filtrar columnas sin nombre (como "Unnamed")
            if variable.startswith("Unnamed"):
                continue
            
            var_var = tk.BooleanVar()
            var_checkbutton = tk.Checkbutton(ventana_variables, text=variable, variable=var_var)
            var_checkbutton.pack()
            var_selection.append((variable, var_var))

        boton_calcular = tk.Button(ventana_variables, text="Calcular Similaridad", command=lambda: calcular_similaridad_desde_seleccion(var_selection))
        boton_calcular.pack()


def select_all_variables(var_selection, var_select_all_var):
    select_all_state = var_select_all_var.get()
    for _, var_var in var_selection:
        var_var.set(select_all_state)
# fucnion para mostrar la guia de usuario 
def mostrar_ayuda():
    ventana_ayuda = tk.Toplevel()
    ventana_ayuda.title("Ayuda - SIMILARIDAD DE ISRAEL LERMAN (PCHIC)")

    texto_ayuda = """SIMILARIDAD DE ISRAEL LERMAN (PCHIC)
IMPORTANTE LEER
Para un adecuado funcionamiento de la aplicación PCHIC es necesario considerar los siguientes pasos:
1. Cargar un archivo de Excel con variables binarias.
2. Seleccionar las variables a usar.
3. Presionar el botón "Calcular Similaridad". Es necesario calcular la matriz 
   de nivel 0 para poder  calcular nodos significativos.
   Es importante cerrar el dendrograma para ver la matriz de nivel 0 en Excel, 
   para ver el gráfico de nodos significativos.
4. Si desea realizar un nuevo análisis con una nueva base de datos binarios,
   es necesario limpiar para que no se sobrepongan los valores.
5. Salir"""

    cuadro_texto = tk.Text(ventana_ayuda, height=20, width=80)
    cuadro_texto.pack(padx=10, pady=10)
    cuadro_texto.insert(tk.END, texto_ayuda)
    cuadro_texto.config(state=tk.DISABLED)  # Hace que el texto no sea editable

    boton_cerrar = tk.Button(ventana_ayuda, text="Cerrar", command=ventana_ayuda.destroy)
    boton_cerrar.pack(pady=10)

def calcular_similaridad_desde_seleccion(var_selection):
    global variables_seleccionadas
    variables_seleccionadas = [variable for variable, var_var in var_selection if var_var.get()]

    if len(variables_seleccionadas) >= 2:
        base_seleccionada = base[variables_seleccionadas]
        calcular_similitud(base_seleccionada)
        
    else:
        resultado_text.delete(1.0, tk.END)
        resultado_text.insert(tk.END, "Seleccione al menos dos variables para calcular la similaridad.\n")
        

import tempfile
import os

def mostrar_matriz_completa():
    # Abrir el archivo CSV con Excel
    os.system('start excel.exe similaridad.xlsx')  # Esto abrirá el archivo CSV con Excel en Windows

# funcion para mostrar niveles de similaridad 
# funcion para mostrar niveles de similaridad 
def mostrar_niveles_similaridad():
    global  ventana_matrices 
   

    # Cerrar la ventana existente si está abierta
    if ventana_matrices:
        ventana_matrices.destroy()

   
 
    ventana_matrices = tk.Toplevel()
    ventana_matrices.title("Matrices de Niveles")

    cuadro_texto = scrolledtext.ScrolledText(ventana_matrices, height=25, width=100)
    cuadro_texto.pack()

    if contenedor_datos_niveles:
        for nivel, maximos, nuevo_vector, vector3, dfss in contenedor_datos_niveles:
            if nivel == 0 :
                
                cuadro_texto.insert(tk.END, "---------------------Niveles de Similaridad----------------------------\n")
                
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    cuadro_texto.insert(tk.END, str(vector3) + "\n")
                    
            else:
                # Verificar si la matriz es toda cero
                if np.all(dfss.values == 0):
                    # Si es la última matriz, no la imprimimos
                    if nivel == len(contenedor_datos_niveles) - 1:
                        break
                    else:
                        continue
                
                cuadro_texto.insert(tk.END, "-----------------------------------------------------------------------------------\n")
                cuadro_texto.insert(tk.END, f" MATRIZ DE NIVEL {nivel}\n")
                
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    cuadro_texto.insert(tk.END, f"{dfss}\n\n")
                cuadro_texto.insert(tk.END, f"VARIABLES {nuevo_vector} = {vector3}\n")
                cuadro_texto.insert(tk.END, f"VARIABLES {nuevo_vector}, VALOR NIVEL : {maximos}\n\n")
                
                # Imprimir variables contenidas en el vector actual
                variables_vector_actual = [var for var in nuevo_vector if not var.startswith(("A", "B", "C"))]
                cuadro_texto.insert(tk.END, f"{vector3} contiene las variables: {', '.join(variables_vector_actual)}\n\n")

                # Imprimir variables contenidas en los vectores anteriores (si existen)
                variables_previas = [var for var in nuevo_vector if var.startswith(("A", "B", "C"))]
                for var_previa in variables_previas:
                    variables_previa_nombres = [var for var in nuevo_vector if var.startswith(var_previa + ",")]
                    cuadro_texto.insert(tk.END, f"{var_previa} contiene las variables: {', '.join(variables_previa_nombres)}\n")

                
                
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    cuadro_texto.insert(tk.END, str(vector3) + "\n\n") # esta opcion sirve para imprimir por partes una matriz 
                                                                   
    else:
        cuadro_texto.insert(tk.END, "No se ha calculado la similitud aún.\n")
        
        
def mostrar_niveles_similaridad1():
    global ventana_matrices1

    if ventana_matrices1:
        ventana_matrices1.destroy()

    ventana_matrices1 = tk.Toplevel()
    ventana_matrices1.title("Niveles de Similaridad")

    cuadro_texto = scrolledtext.ScrolledText(ventana_matrices1, height=25, width=100)
    cuadro_texto.pack()

    if contenedor_datos_niveles:
        for nivel, maximos, nuevo_vector, vector3, dfss in contenedor_datos_niveles:
            cuadro_texto.insert(tk.END, f"---------------------Nivel {nivel}----------------------------\n")
            cuadro_texto.insert(tk.END, f"VARIABLES: {nuevo_vector}\n")
            cuadro_texto.insert(tk.END, f"VALOR DEL NIVEL: {maximos}\n")

            if nivel > 0:
                cuadro_texto.insert(tk.END, f"NUEVA VARIABLE: {vector3}\n")

                # Imprimir variables contenidas en el vector actual
                variables_vector_actual = [var for var in nuevo_vector if not var.startswith(("A", "B", "C", "V")) or (var.startswith("V") and len(var) > 2)]
                if variables_vector_actual:
                    cuadro_texto.insert(tk.END, f"{vector3} contiene las variables: {', '.join(variables_vector_actual)}\n")

                # Imprimir variables contenidas en los vectores anteriores (si existen)
                variables_previas = [var for var in nuevo_vector if var.startswith(("A", "B", "C")) or (var.startswith("V") and len(var) > 2)]
                for var_previa in variables_previas:
                    variables_previa_nombres = [var for var in nuevo_vector if var.startswith(var_previa + ",")]
                    if variables_previa_nombres:
                        cuadro_texto.insert(tk.END, f"{var_previa} contiene las variables: {', '.join(variables_previa_nombres)}\n")

            cuadro_texto.insert(tk.END, "\n")

    else:
        cuadro_texto.insert(tk.END, "No se ha calculado la similitud aún.\n")
# funcion para  crear el boton de nodos signifcativos
def calcular_nodos_significativos():
    if contenedor_lista_matriz:
        ordenar_nodos(contenedor_lista_matriz)
        grafica_omega(contenedor_v)
    else:
        resultado_text.delete(1.0, tk.END)
        resultado_text.insert(tk.END, "Primero calcula la similaridad.\n")        

# funcion para  crear el boton de nodos signifcativos
def calcular_nodos_significativos():
    if contenedor_lista_matriz:
        ordenar_nodos(contenedor_lista_matriz)
        grafica_omega(contenedor_v)
    else:
        resultado_text.delete(1.0, tk.END)
        resultado_text.insert(tk.END, "Primero calcula la similaridad.\n")
# Función para limpiar las variables que se llenan al realizar el análisis de la similitud de Lerman.





def borrar_valores():
    resultado_text.delete(1.0, tk.END)
    contenedor_valor.clear()
    contenedor_v.clear()
    contenedor_valor.clear()
    contenedor_de_la_matriz.clear()
    contendor_diccionario.clear()
    contenedor_f.clear()
    contenedor_sumaI.clear()
    contenedor_rkk.clear()
    contenedor_skk.clear()
    contenedor_lista_matriz.clear()
    fi.clear()
    Ii.clear()
    yy.clear()
    xx.clear()
    contenedor_datos_niveles.clear()

    # Cerrar la ventana de niveles de similaridad
    global ventana_matrices
    if ventana_matrices:
        ventana_matrices.destroy()
        ventana_matrices = None
        
    global ventana_matrices1
    if ventana_matrices1:
        ventana_matrices1.destroy()
        ventana_matrices1 = None    


    

ventana = tk.Tk()
ventana.title("Calcular Similaridad y Nodos Significativos")
ventana_matrices = None 
ventana_matrices1 = None 
cuadro_texto_niveles = None


#  # Creación del menú desplegable.

menubar = tk.Menu(ventana)
archivo_menu = tk.Menu(menubar, tearoff=0)
archivo_menu.add_command(label="Seleccionar Archivo csv", command=seleccionar_archivo)
archivo_menu.add_command(label=" Nodos Significativos", command=calcular_nodos_significativos)
archivo_menu.add_command(label="Limpiar", command=borrar_valores)
archivo_menu.add_command(label="Salir", command=salir)
menubar.add_cascade(label="Similaridad Lerman", menu=archivo_menu)

ver_menu = tk.Menu(menubar, tearoff=0)
ver_menu.add_command(label="Ver Matriz Completa", command=mostrar_matriz_completa)
ver_menu.add_command(label="Ver Matriz de Niveles de Similaridad", command=mostrar_niveles_similaridad)
ver_menu.add_command(label="Ver valores del de Similaridad", command=mostrar_niveles_similaridad1)
menubar.add_cascade(label="Ver", menu=ver_menu)

Ayuda_menu = tk.Menu(menubar, tearoff=0)
Ayuda_menu.add_command(label="Ayuda", command=mostrar_ayuda)
menubar.add_cascade(label="Ayuda", menu=Ayuda_menu)

ventana.config(menu=menubar)

resultado_text = scrolledtext.ScrolledText(ventana, height=45, width=100)
resultado_text.pack(pady=15)

ventana.mainloop()