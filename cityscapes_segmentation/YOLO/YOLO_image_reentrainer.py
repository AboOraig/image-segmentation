# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 09:53:16 2025

@author: maboorai279

Ce code utilise le modèle de segmentation YOLOv8 pour entraîner un modèle personnalisé sur le jeu de données Cityscapes.
Après l'entraînement, le modèle est utilisé pour effectuer une inférence sur une image test.
Les résultats incluent l'affichage de l'image annotée et l'impression des boîtes englobantes,
des classes détectées et des scores de confiance.
"""

from ultralytics import YOLO

def main():
    # Charger le modèle préentraîné soit yolov8n-seg ou yolo11n-seg
    #model = YOLO("yolov8n-seg.pt")
    model = YOLO("yolo11n-seg.pt")

    # Afficher les infos du modèle
    model.info()

    # Entraîner le modèle
    #results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640)
    results = model.train(
        data="D:/IRIS/A2/Projet/datasets_seg2/cityscapes/cityscapes.yaml",
        epochs=100,
        imgsz=640,
        val=False,
        device=0
    )

    # Inférence sur une image
    results = model("test2.png")
    results[0].show()

    # Affichage des résultats
    for r in results:
        print(r.boxes)  # Coordonnées des boîtes
        print(r.names)  # Noms des classes détectées
        print(r.probs)  # Scores de confiance

if __name__ == "__main__":
    main()