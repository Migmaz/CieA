"""
MAIN
Chef d’orchestre du projet.
 
Rôle :
- Boucle principale du robot
- Lit les données capteurs (sensor.py)
- Demande à control.py quel comportement utiliser
- Exécute le behavior correspondant (behaviors.py)
- Envoie la commande aux moteurs (actuation.py)
"""
 
# =================================================
# Importation de librairy générale
# =================================================
import time
import numpy as np
from pynput import keyboard
 
# =================================================
# Importation de librairy Périphérique
# =================================================
import board
import busio
from digitalio import DigitalInOut
 
from adafruit_pca9685 import PCA9685
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import(
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_ROTATION_VECTOR,
)
 
 
# =================================================
# Importation de fonctions
# =================================================
 
from FollowGap import FTG, FSM, tool, sensor, behaviors, mapping
 
# =================================================
# Initialisation des capteurs
# =================================================
 
 
 
 
# PWM
i2c_PWM = board.I2C()
pca = PCA9685(i2c_PWM)
pca.frequency = 60
 
# =================================================
# Sélection mode de contrôle
# =================================================
 
print("Veuillez choisir le mode de contrôle pour le rover : ")
print(" 1 - Mode autonome : Follow the gap")
print(" 2 - Mode manuel : Clavier AWSD")
choix_mode = input("Entrer le choix du mode : ")
 
if choix_mode == "1" :
    print("Contrôle autonome sélectionné")
    # =================================================
    # Boucle contrôle autonome
    # =================================================
   
    # Initialisation du IMU
    i2c_IMU = busio.I2C(board.SCL, board.SDA, frequency=100000)
    bno = BNO08X_I2C(i2c_IMU)
    bno.enable_feature(BNO_REPORT_ACCELEROMETER)
    bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
 
   
    # Boucle principale
 
    state = FSM.RobotState()
 
    while True:
 
        # 1. Capteurs
        a, yaw = sensor.IMU(bno)
 
        # 2. Objectif
        Position_rover = mapping.update_position(x, y, yaw, a * 0.01, 0.01) #Linear_speed = acceleration * temps (dt)
        theta_goal = tool.theta_goal(Position_rover, Pg, yaw)
 
        # 3. Lidar
        scan_clean = tool.preprocess_lidar(scan)
        pts = tool.trans_to_rover(scan_clean, pitch, translation)
        pts = tool.filter_ground(pts)
        scan_eff, scan_true = tool.compute_scan(pts)
 
        # FSM → scan réel
        behavior = FSM.update_state(
            scan_eff, theta_goal, Position_rover, Pg, state
        )
 
 
        # 5. Behavior
        if behavior == "NAVIGATE":
            cmd = behaviors.navigate(theta_goal)
 
        elif behavior == "SCAN":
            cmd = behaviors.scan_behavior(scan_eff)
 
        elif behavior == "RETOUR_BASE":
 
            theta_retour = tool.theta_goal(Position_rover, (0,0), yaw) # La position de départ est ici initialisée à (0, 0), mais peut être ajustée si nécessaire.
            cmd = behaviors.retour_base(theta_retour)
 
        elif behavior == "STOP":
            cmd = behaviors.stop()
 
        elif behavior in ["ESCAPE", "CUL-DE-SAC"]:
            cmd = behaviors.escape(scan_eff)
       
        else:
            cmd = {"linear": 0.0, "angular": 0.0}
 
 
        # 6. ACTION (CRUCIAL)
        tool.apply_command(pca, cmd)
 
        time.sleep(0.01)
       
 
 
 
   
   
else :
    print("Contrôle manuel sélectionner")
    # =================================================
    # Boucle contrôle manuel
    # =================================================
   
    # Initialisation de la lecture du clavier
    def on_press(key, injected):
        try:
            None
        except AttributeError:
            print('special key {} pressed'.format(
                key))
 
    def on_release(key, injected):
        if key == keyboard.Key.esc:
            # Stop listener
            return False
 
    # Collect events until released
    listener = keyboard.Listener()
    listener.start()
   
    # Boucle principale
    while True:
        with keyboard.Events() as events:
        # Block at most one second (à modifier pour fit la boucle)
            event = events.get(1.0)
            if event is None:
                print('You did not press a key within one second')
            else:
                print(event.key)
               
                # Garde en mémoire la dernier touche
                key_temp = event.key
                # Ferme le programme avec la touche escape
                if event.key == keyboard.Key.esc : break
 