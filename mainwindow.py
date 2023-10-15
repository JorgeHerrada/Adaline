from PyQt5.QtWidgets import QMainWindow
from ui_mainwindow import Ui_MainWindow     # importamos la clase que define la UI
from PyQt5.QtCore import pyqtSlot
from perceptron import Perceptron
from PyQt5 import QtGui
import numpy as np

class MainWindow(QMainWindow):
    entradas = []
    salidas = []

    def __init__(self):
        super(MainWindow,self).__init__() # inicializa desde clase padre

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.btnAgregar.clicked.connect(self.click_agregar)
        self.ui.btnClasificar.clicked.connect(self.click_clasificar)

        # Creamos neurona, entrada y learning rate
        self.neuron = Perceptron()


    @pyqtSlot()
    def click_agregar(self):
        try:
            # Primer punto? limpiamos plot
            if len(self.entradas) == 0:
                self.neuron.clear()

            # hay texto en las cajas?
            if self.ui.txtX1.text() != "" and self.ui.txtX2.text() != "":
                # guardamos guardamos entradas (x1,x2) y salidas (y)
                self.entradas.append([float(self.ui.txtX1.text()),float(self.ui.txtX2.text())])
                print("entradas: ",self.entradas)
                if self.ui.checkBox.checkState():
                    self.salidas.append(1)
                    
                    # PLOTTEAR
                    self.neuron.graficador.setPunto(self.entradas[-1][0],self.entradas[-1][1],1)
                else:
                    self.salidas.append(0)

                    # PLOTTEAR
                    self.neuron.graficador.setPunto(self.entradas[-1][0],self.entradas[-1][1],0)
                # print("Salidas: ", self.salidas)
            else:
                print("Entrada invalida")
            
            # actualizar UI
            # print("apunto de guardarActualizar")
            self.neuron.guardarActualizar(self.ui)

            # limpiar 
            self.ui.txtX1.setText("")
            self.ui.txtX2.setText("")

        except ValueError:
            print("¡Error en la entrada! - ",ValueError)

    @pyqtSlot()
    def click_clasificar(self):
        
        if self.ui.btnLineal.isChecked():
            self.neuron.funcionSeleccionada = "lineal"
            # print("LINEAL ",self.ui.btnLineal.isChecked())
        elif self.ui.btnLogistica.isChecked():
            # print("LOGISTICA ", self.ui.btnLogistica.isChecked())
            self.neuron.funcionSeleccionada = "logistica"
        else:
            # print("TANGENCIAL ",self.ui.btnTangencial.isChecked())
            self.neuron.funcionSeleccionada = "tangencial"
            self.salidas = [-1 if item == 0 else item for item in self.salidas]

        # Creamos matriz de entradas
        X = np.zeros(shape=(2,len(self.salidas)))
        # print("X: ", X)
        for i in range(len(self.entradas[0])):
            for j in range(len(self.salidas)):
                X[i,j] = self.entradas[j][i]
        print("X: ", X)

        # Matriz de salidas deseadas (1 por cada par de entradas)
        Y = np.array(self.salidas)
        print("Y: ", Y)

        # neurona aprende e imprime resultados
        print("Pre entrenamiento: ",self.neuron.predict(X))
        self.neuron.fit(X, Y, self.ui)
        print("Post entrenamiento: ",self.neuron.predict(X))

        # limpiamos
        self.entradas = []
        self.salidas = []
        

        
 