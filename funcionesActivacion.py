from math import exp,tanh,cos

# funcion de activacion lienal


def lineal(num):
    return num


def logistica(num):
    return 1/(1+(exp(-num)))
    # 1/(1+exp(2))

def tangencial(num):
    return tanh(num)


def linealDerivada(num):
    return 1

def logisticaDerivada(num):
    return exp(-num)/((1+exp(-num))**2)

def tangencialDerivada(num):
    return (1/cos(num))**2
