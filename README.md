# AER1110-A

**Projet intégrateur AER1110 – Équipe A**  

Ce dépôt contient le code pour un prototype d'un rover lunaire de **modélisation 3D** dans le cadre du cours AER1110. 

---

## Table des matières

- [Description](#description) 
- [Fonctionnalités](#fonctionnalités)  
- [Prérequis](#prérequis)  
- [Installation](#installation)

---

## Description

Le prototype sera tester sur un terrain simulant celui lunaire avec comme tâches :

- Naviguer de manière **autonome** vers un objets précis et revenir au point de départ
- Générer un **modèle 3D** de l'objet
 
---

## Fonctionnalités

1. **Navigation autonome**  
   - L'algorithme follow the gap a été implementer afin d'éviter les obstacles  

2. **Modélisation 3D**  
   - Un LiDAR 2D inclinée s'occupe de la modélisation avec l'aide de Open3D 

---

## Prérequis

Avant d’utiliser ce code, assurez-vous d’avoir installé :  

- Python ≥ 3.10 (compatible jusqu’à 3.12 pour certaines librairies)  
- Librairies Python :
  ```
  numpy
  gpiozero
  adafruit-circuitpython-bno08x
  open3d  # pour la modélisation 3D
  ```

---

## Installation

1. Cloner le repo
```
git clone https://github.com/votre-utilisateur/AER1110-A.git
cd AER1110-A
```
2. Crée un environnement virtuel
```
python3 -m venv venv
source venv/bin/activate  # sur Linux/macOS
```
3. Installer les dépendances
```
pip install -r requirements.txt
```
