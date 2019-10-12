import math

def main():
    print("PROPAGACION Y RETROPROPAGACION")
    factorAprendizaje = float(input("Factor de aprendizaje: ")) #Factor de aprendizaje
    #factorAprendizaje = 0.5
    numCapas = int(input("Numero de capas: ")) #Numero de capas
    #numCapas = 2
    salidas = [[] for i in range(0,numCapas+1)] #Creamos el contenedor de salidas
    bias = [[] for i in range(0,numCapas)] #Creamos el contenedor de bias
    pesos = [[] for i in range(0,numCapas)] #Creamos el contenedor de pesos
    errores = [None for i in range(0,numCapas)] #Creamos el contenedor de errores
    celulas = [0 for i in range(0,numCapas)] #Creamos el contenedor del numero de celulas por capa
    #Inicializacion de las entradas
    numEntradas = int(input("Numero de entradas: "))
    #numEntradas = 2
    entradas = []
    for i in range(0,numEntradas):
        entradas.append(float(input("Valor de la entrada " + str(i+1) + ": ")))
    salidas[0].append(entradas)
    #Numero de celulas por capa
    for i in range(0,numCapas):
        celulas[i] = int(input("Numero de celulas en la capa " + str(i+1) + ": "))
    #Inicializacion de las bias
    for i in range(0,numCapas):
        aux = []
        for j in range(0,celulas[i]):
            aux.append(float(input("Valor de la bia " + str(j+1) + " en la capa " + str(i+1) + ": ")))
        bias[i].append(aux)
    #Inicializacion de los pesos
    for i in range(0,numCapas):
        for j in range(0,numEntradas):
            aux = []
            for k in range(0,celulas[i]):
                aux.append(float(input("Valor del peso entre la salida " + str(j+1) + " de la capa " + str(i) + " y la celula " + str(k+1) + " de la capa " + str(i+1) + ": ")))
            pesos[i].append(aux)
    #Incializacion de las salidas deseadas
    deseados = [[] for i in range(0,celulas[len(celulas)-1])] #Creamos el contenedor de las salidas deseadas
    for i in range(0,len(deseados)):
        deseados[i] = [float(input("Valor deseado para la salida " + str(i+1) + ": "))]
    iteraciones = int(input("Numero de iteraciones a realizar: "))
    i = 1
    while(i <= iteraciones):
        print("-- VALORES TRAS LA PROPAGACION " + str(i) + " --")
        propagacion(salidas, pesos, bias, celulas, errores, deseados)
        mostrarBonito(salidas, pesos, bias, errores)
        print("-- VALORES TRAS LA RETROPROPAGACION " + str(i) + " --")
        retropropagacion(factorAprendizaje, salidas, errores, pesos, bias)
        mostrarBonito(salidas, pesos, bias, errores)
        i += 1

def pesoNuevo(pesoActual, factorAprendizaje, salidaAntigua, errorActual):
    return sumarMatrices(pesoActual, multiplicarMatrizPorEscalar(factorAprendizaje, multiplicarMatrices(matrizTraspuesta(salidaAntigua),errorActual)))

def biaNueva(biaActual,factorAprendizaje,errorActual):
    return sumarMatrices(biaActual, multiplicarMatrizPorEscalar(factorAprendizaje,errorActual))

def salida(salidaCapaAnterior, pesosCapaActual, biaCapaActual):
    return funcionSigmoidal(sumarMatrices(multiplicarMatrices(salidaCapaAnterior,pesosCapaActual),biaCapaActual))

def errorCelulaOculta(errorCapaSuperior, pesosCapaSuperior, salida):
    return multiplicarMatricesPuntoPorPunto(multiplicarMatrices(errorCapaSuperior,matrizTraspuesta(pesosCapaSuperior)),funcionSigmoidalDerivada(salida))

def errorCelulaVisible(deseado, salida):
    return multiplicarMatricesPuntoPorPunto(restarMatrices(deseado,salida),funcionSigmoidalDerivada(salida))

def mostrarBonito(salidas, pesos, bias, errores):
    print("*Entradas*")
    j = 1
    for i in salidas[0][0]:
        print("-> Entrada " + str(j) + ": " + str(i))
        j += 1
    print("*Salidas*")
    j = 1
    for i in range(1,len(salidas)):
        print("-> Capa " + str(j))
        k = 1
        for salida in salidas[i][0]:
            print("---> Celula " + str(k) + ": " + str(salida))
            k += 1
        j += 1
    print("*Pesos*")
    j = 1
    for peso in pesos:
        print("-> Capa " + str(j))
        k = 1
        for fila in peso:
            i = 1
            for columna in fila:
                print("----> Celula " + str(k) + " (Capa " + str(j-1) + ") -- Celula " + str(i) + " (Capa " + str(j) + "): " + str(columna))
                i += 1
            k += 1
        j += 1
    print("*Bias*")
    j = 1
    for i in range(0,len(bias)):
        print("-> Capa " + str(j))
        k = 1
        for bia in bias[i][0]:
            print("---> Celula " + str(k) + ": " + str(bia))
            k += 1
        j += 1
    print("*Errores*")
    j = 1
    for i in range(0,len(errores)):
        print("-> Capa " + str(j))
        k = 1
        for error in errores[i][0]:
            print("---> Celula " + str(k) + ": " + str(error))
            k += 1
        j += 1

#################################################################################################################
# FUNCIONES PRINCIPALES #########################################################################################
#################################################################################################################

def propagacion(salidas, pesos, bias, celulas, errores, deseados):
    #Calcular las salidas de las celulas de cada capa
    for i in range(0,len(celulas)):
        salidas[i+1] = salida(salidas[i], pesos[i], bias[i])
    #Calcular los errores de las capas
    errores[len(celulas)-1] = errorCelulaVisible(deseados,salidas[len(celulas)]) #Capa final
    i = len(celulas)-2 #Capas ocultas
    while(i >= 0):
        errores[i] = errorCelulaOculta(errores[i+1],pesos[i+1],salidas[i+1])
        i = i - 1

def retropropagacion(factorAprendizaje, salidas, errores, pesos, bias):
    #Calculo de nuevos pesos
    for i in range(0,len(pesos)):
        pesos[i] = pesoNuevo(pesos[i], factorAprendizaje, salidas[i], errores[i])
    #Calculo de nuevas bias
    for i in range(0,len(bias)):
        bias[i] = biaNueva(bias[i],factorAprendizaje,errores[i])

#################################################################################################################
# FUNCIONES DE ACTIVACION #######################################################################################
#################################################################################################################

def funcionSigmoidal(matriz):
    matrizFilas = len(matriz)
    matrizColumnas = len(matriz[matrizFilas-1])
    matrizResultado = []
    for fila in range(0,matrizFilas):
        filaAux = []
        for columna in range(0,matrizColumnas):
            filaAux.append(round(1/(1+math.exp((-1) * matriz[fila][columna])),3))
        matrizResultado.append(filaAux)
    return matrizResultado
def funcionSigmoidalDerivada(salida):
    resul = []
    for i in range(0,len(salida)):
        aux = []
        for j in range(0,len(salida[i])):
            aux.append(1-salida[i][j])
        resul.append(aux)
    return multiplicarMatricesPuntoPorPunto(salida,resul)

#################################################################################################################
# OPERACIONES BASICAS CON MATRICES ##############################################################################
#################################################################################################################

def multiplicarMatrices (matrizA, matrizB):
    filasMatrizA = len(matrizA)
    columnasMatrizA = len(matrizA[0])
    filasMatrizB = len(matrizB)
    columnasMatrizB = len(matrizB[0])
    matrizSolucion = [[0 for fila in range(columnasMatrizB)] for columna in range(filasMatrizA)]
    for i in range(filasMatrizA):
        for j in range(columnasMatrizB):
            for k in range(columnasMatrizA):
                matrizSolucion[i][j] += round(matrizA[i][k] * matrizB[k][j],3)
    return matrizSolucion

def multiplicarMatrizPorEscalar(escalar, matriz):
    matrizFilas = len(matriz)
    matrizColumnas = len(matriz[matrizFilas-1])
    matrizSolucion = []
    for fila in range(0,matrizFilas):
        filaAux = []
        for columna in range(0,matrizColumnas):
            filaAux.append(round(matriz[fila][columna]*escalar,3))
        matrizSolucion.append(filaAux)
    return matrizSolucion

def multiplicarMatricesPuntoPorPunto(matrizA, matrizB):
    filasMatrizA = len(matrizA)
    filasMatrizB = len(matrizB)
    columnasMatrizA = len(matrizA[filasMatrizA-1])
    columnasMatrizB = len(matrizB[filasMatrizB-1])
    matrizC = []
    for fila in range(0,filasMatrizA):
        filaAux = []
        for columna in range(0,columnasMatrizA):
            filaAux.append(round(matrizA[fila][columna]*matrizB[fila][columna],3))
        matrizC.append(filaAux)
    return matrizC

def sumarMatrices(matrizA, matrizB):
    filasMatrizA = len(matrizA)
    filasMatrizB = len(matrizB)
    columnasMatrizA = len(matrizA[filasMatrizA-1])
    columnasMatrizB = len(matrizB[filasMatrizB-1])
    matrizC = []
    for fila in range(0,filasMatrizA):
        filaAux = []
        for columna in range(0,columnasMatrizA):
            filaAux.append(matrizA[fila][columna]+matrizB[fila][columna])
        matrizC.append(filaAux)
    return matrizC

def restarMatrices(matrizA, matrizB):
    filasMatrizA = len(matrizA)
    filasMatrizB = len(matrizB)
    columnasMatrizA = len(matrizA[filasMatrizA-1])
    columnasMatrizB = len(matrizB[filasMatrizB-1])
    matrizC = []
    for fila in range(0,filasMatrizA):
        filaAux = []
        for columna in range(0,columnasMatrizA):
            filaAux.append(matrizA[fila][columna]-matrizB[fila][columna])
        matrizC.append(filaAux)
    return matrizC

def matrizTraspuesta(matriz):
    matrizFilas = len(matriz)
    matrizColumnas = len(matriz[matrizFilas-1])
    matrizResultado = []
    for columna in range(0,matrizColumnas):
        filaAux = []
        for fila in range(0,matrizFilas):
            filaAux.append(matriz[fila][columna])
        matrizResultado.append(filaAux)
    return matrizResultado

#################################################################################################################
# MAIN ##########################################################################################################
#################################################################################################################

if __name__=="__main__":
    main()
