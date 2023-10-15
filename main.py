from PyQt5.QtWidgets import QApplication
from mainwindow import MainWindow   # clase creada para manejar la ventana
import sys


if __name__ == "__main__":
    app = QApplication(sys.argv)    # crea app de QT
    window = MainWindow()           # crea ventana 
    window.show()                   # crea 

    sys.exit(app.exec_())





