import numpy as np
from PyQt5 import QtTest
import matplotlib.pyplot as plt
from graficador import Graficafor
from PyQt5 import QtGui
import funcionesActivacion as func


class Perceptron:
    # defininmos limite de de epocas para evitar
    # bucles infinitos en caso de que los puntos
    # insertados no sean linealmente separables
    EPOCH = 100
    LEARN_RATE = 0.3  # learning rate
    N_INPUT = 2  # dimension de los patrones de entrenamiento

    funcionSeleccionada = ""

    contEpoch = 0  # contador de epocas

    # Constructor toma numero de inputs y learning rate
    def __init__(self, n_input=N_INPUT, learning_rate=LEARN_RATE):

        # inicializacion random en funcion clear
        self.clear(n_input, learning_rate)
        # instanciamos graficador
        self.graficador = Graficafor()

    # funcion de activacion
    def f_activacion(self, num):
        # print("LA funcion seleccionada es: {}".format(self.funcionSeleccionada))
        if self.funcionSeleccionada == "lineal":
            # print("LINEAL: f({})".format(num))
            return func.lineal(num)
        elif self.funcionSeleccionada == "logistica":
            # print("LOGISTICA: f({})".format(num))
            return func.logistica(num)
        else:
            # print("TANGENCIAL: f({})".format(num))
            return func.tangencial(num)

    # funcion de activacion derivada
    def f_activacionDerivada(self, num):
        if self.funcionSeleccionada == "lineal":
            return func.linealDerivada(num)
        elif self.funcionSeleccionada == "logistica":
            return func.logisticaDerivada(num)
        elif self.funcionSeleccionada == "tangencial":
            return func.tangencialDerivada(num)

    # Entrega un vector de salidas, dada una matriz de
    # entradas para la neurona
    def predict(self, X):

        # p es el numero de columnas en la matriz X
        p = X.shape[1]

        # y_est guardará la salidas, se inicializa
        # como vector de p dimensiones con 0s
        y_est = np.zeros(p)

        # iteramos por cada solucion a generar
        for i in range(p):
            # calculamos salidas
            y_est[i] = np.dot(self.w, X[:, i]) + self.b
            # asignamos valor binario según la funcion de activacion
            y_est[i] = self.f_activacion(y_est[i])

        # retornamos vector con las salidas binarias
        return y_est

    # Realiza aprendizaje en epocas, actualiza
    # puntos y division en grafica
    def fit(self, X, Y, ui, epoch=EPOCH):

        # p es el numero de conjuntos de entrada (patron)
        p = X.shape[1]
        # lista para comparar estimaciones con resultados esperados
        estimaciones = []
        self.contEpoch = 0

        # iteramos por cada epoca
        for _ in range(epoch):
            # resetea estimaciones y aumenta contador
            estimaciones = []
            self.contEpoch += 1

            # iteramos por cada patron de entrenamiento
            for i in range(p):

                # calculamos salida dado el patron actual
                # reshape para asegurar que tenemos vector columna
                y_est = self.predict(X[:, i].reshape(-1, 1))
                estimaciones.append(y_est)  # guardamos estimacion

                # actualizacion de peso y bias basado en el error
                self.w = self.w + self.eta * \
                    (Y[i] - y_est) * X[:, i] * self.f_activacionDerivada(y_est)
                self.b = self.b + self.eta * \
                    (Y[i] - y_est) * self.f_activacionDerivada(y_est)

                # actualizacion en UI
                ui.txtW1.setText(str(round(self.w[0], 6)))
                ui.txtW2.setText(str(round(self.w[1], 6)))
                ui.txtTheta.setText(str(-round(self.b[0], 6)))

            # limpiar
            plt.cla()

            # plottear puntos a color
            self.graficador.plotMatrix(X, estimaciones)

            # plottear linea
            self.graficador.drawDivision([self.punto(self.w[0], self.w[1], -self.b, -5),
                                          self.punto(self.w[0], self.w[1], -self.b, 5)],)

            # actualizar
            self.guardarActualizar(ui)

            # se logró el objetivo?
            if self.aprendizajeTerminado(Y, estimaciones):
                break

            # retraso para visualizar
            QtTest.QTest.qWait(100)

        print("Epocas: ", self.contEpoch)

    # ¿todas las estimaciones son iguales a las salidas esperadas?
    def aprendizajeTerminado(self, Y, estimaciones):
        print("Y: ", Y)
        print("estimaciones: ", estimaciones)

        errorActual = 0
        for i in range(len(estimaciones)):
            errorActual = errorActual + (Y[i] - estimaciones[i])**2

        errorPromedio = errorActual/len(estimaciones)
        print("Error promedio: ", errorPromedio)

        if errorPromedio < self.errorObjetivo:
            return True
        else:
            return False

    # calcular punto para pendiente
    def punto(self, w1, w2, teta, x):
        if w2 == 0:
            print("No se puede dividir entre cero, cambia el valor de W2")
            return

        m = -1*(w1/w2)
        c = teta/w2
        y = (m*x) + c
        return y

    # calcular pendiente de recta que separa 1s y 0s
    def calcularPendiente(self, columna):
        # linea para dividir
        limLinea = [-5, 5]
        p1 = self.punto(self.w[0, columna],
                        self.w[1, columna], -self.b, limLinea[0]),
        p2 = self.punto(self.w[0, columna],
                        self.w[1, columna], -self.b, limLinea[0]),

        return p1, p2

    # save fig and update UI
    def guardarActualizar(self, ui):
        plt.savefig("prueba.png")
        ui.label.setPixmap(QtGui.QPixmap("prueba.png"))

    # limpia puntos viejos y reinicia pesos y bias
    def clear(self, n_input=2, learning_rate=0.1):
        # plt.clf()
        plt.cla()

        # inicializamos los pesos "w" con un vector de
        # dimension "n_input" con rango [-1,1] random
        self.w = -1 + (1 - (-1)) * np.random.rand(n_input)
        # bias random con rango [-1,1]
        self.b = -1 + (1 - (-1)) * np.random.rand()
        self.eta = learning_rate
        self.errorObjetivo = 0.001
