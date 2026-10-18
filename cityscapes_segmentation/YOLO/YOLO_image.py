# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 09:52:56 2025

@author: maboorai279

Ce script applique un modèle de segmentation YOLO pré-entraîné sur une image fixe.
Il permet de visualiser les résultats de l'inférence, notamment les boîtes englobantes,
les classes détectées et les scores de confiance associés.
Le modèle utilisé peut être soit yolov8n-seg, soit yolo11n-seg selon le besoin.
"""

from ultralytics import YOLO

# Charger le modèle préentraîné soit yolov8n-seg ou yolo11n-seg
model = YOLO("yolov8n-seg.pt")
#model = YOLO("yolo11n-seg.pt")

# Afficher les infos du modèle
model.info()

# Entraîner le modèle
#results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Inférence sur une image
results = model("test2.png")

for r in results:
    print(r.boxes)  # Affiche les coordonnées des boîtes englobantes
    print(r.names)  # Affiche les noms des classes détectées
    print(r.probs)  # Affiche les scores de confiance

results[0].show()