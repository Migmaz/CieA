"""
SENSOR
Gestion des capteurs.

Rôle :
- Lire les données du LiDAR (scan)
- Lire les données IMU (yaw / orientation)
- (Optionnel) Estimer la position (odometry)

Sorties standard :
- scan : liste de distances
- yaw : orientation du robot
"""

#Code permettant de lire les données du IMU

#!/usr/bin/python3
# -*- coding: utf-8 -*-

# GUI
from PyQt5 import QtWidgets, QtCore, QtGui
from pyqtgraph import PlotWidget
import pyqtgraph as pg

import sys
import board
import adafruit_fxos8700
import adafruit_fxas21002c
import imufusion
import numpy
import collections

#Rôle ligne 34 à 36: 'instances of i2c communication and sensors are created' (explication à 8min13 de la vidéo, lien à la fin du code)
i2c = board.I2C()
fxos = adafruit_fxos8700.FXOS8700(i2c)
fxas = adafruit_fxas21002c.FXAS21002C(i2c, gyro_range = adafruit_fxas21002c.GYRO_RANGE_2000DPS)

#Rôle des lignes 39 à 46: Les différents paramètres de l'algorithme IMU Fusion (cette algorithme quoi) sont définis.
time_interval = 0.01
sample_rate = int(1 / time_interval)

ahrs = imufusion.Ahrs()
ahrs.settings = imufusion.Settings(10,  # gain
                                   10,  # acceleration rejection
                                   20,  # magnetic rejection
                                   5 * sample_rate)  # rejection timeout

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)
        self.i = 0

        #Rôle des lignes 58 à 61: 'The dequeue of size 20 is defined' (explication à 8min40 de la vidéo, lien à la fin du code)
        self.time_data = collections.deque([],20)
        self.euler_x = collections.deque([],20)
        self.euler_y = collections.deque([],20)
        self.euler_z = collections.deque([],20)

        #Rôle lignes 64 à 74: Définir plusieurs paramètres pour le graphique (ce code inclut également l’affichage d’un graphique).
        self.graphWidget.setBackground(QtGui.QColor(255, 255, 255, 255))
        self.graphWidget.setYRange(-180, 180)
        self.graphWidget.setLabels(bottom = "time (s)", left = "euler angles deg)")
        self.graphWidget.getAxis('left').setTickSpacing(major = 90, minor = 10)
        self.graphWidget.showGrid(x=True,y=True)
        pen_x = pg.mkPen(color=(255, 0, 0), width=2, style = QtCore.Qt.SolidLine)
        pen_y = pg.mkPen(color=(0, 255, 0), width=2, style = QtCore.Qt.SolidLine)
        pen_z = pg.mkPen(color=(0, 0, 255), width=2, style = QtCore.Qt.SolidLine)
        self.data_line_x = self.graphWidget.plot(self.time_data, self.euler_x, pen=pen_x)
        self.data_line_y = self.graphWidget.plot(self.time_data, self.euler_y, pen=pen_y)
        self.data_line_z = self.graphWidget.plot(self.time_data, self.euler_z, pen=pen_z)

        #Rôle des lignes 77 à 80: Définir la fréquence de mise à jour des données.
        self.timer = QtCore.QTimer()
        self.timer.setInterval(int(time_interval*1000)) #Remet à jour les données du IMU à chaque 10 miliseconde
        self.timer.timeout.connect(self.update_plot_data)
        self.timer.start()

    def update_plot_data(self):

        self.time_data.append(self.i*time_interval)

        #Rôle des lignes 87 à 89: Récupérer les données du gyroscope, accéléromètre et magnétomètre, puis les convertir en numpy array
        gyroscope = numpy.array(fxas.gyroscope)
        accelerometer = numpy.array(fxos.accelerometer)
        magnetometer = numpy.array(fxos.magnetometer)

        #Rôle des lignes 92 à 104: La fusion des données s'exécute et le programme rajoute aux dequeue les angles calculés. L'angle qui nous intérresse est euler[2] qui représente la rotation autour de l'axe des z (comme tourner la tête à droite ou à gauche) et qui nous permet de savoir dans quelle direction se trouve l'objet à analyser par rapport au rover. Pour plus d'informations, voir la vidéo à 9min15
        ahrs.update(gyroscope, accelerometer, magnetometer, 1 / 100)

        euler = ahrs.quaternion.to_euler()
        print(f'euler: {euler}')

        self.euler_x.append(euler[0])
        self.euler_y.append(euler[1])
        self.euler_z.append(euler[2])

        self.data_line_x.setData(self.time_data , self.euler_x)
        self.data_line_y.setData(self.time_data , self.euler_y)
        self.data_line_z.setData(self.time_data , self.euler_z)
        self.i += 1

app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
w.show()
sys.exit(app.exec_())

#Source: https://www.youtube.com/watch?v=yW22igLjkjY&list=PLw_ISvz6MU4w2zelZ8BhB5e3-9npV213G&index=3
