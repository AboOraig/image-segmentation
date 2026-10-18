# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 11:35:26 2025

@author: maboorai279

Ce code applique le module de segmentation YOLOv8 sur une vidéo en entrée.
Il utilise un modèle pré-entraîné (yolov8n-seg ou yolo11n-seg) pour détecter et segmenter les objets image par image,
puis génère une nouvelle vidéo annotée avec les résultats de la détection.
Enfin, il affiche les informations sur les objets détectés, notamment les boîtes englobantes,
les noms des classes et les scores de confiance.
"""

import cv2
from ultralytics import YOLO

# Charger un modèle pré-entraîné soit yolov8n-seg ou yolo11n-seg
model = YOLO("yolov8n-seg.pt")
#model = YOLO("yolo11n-seg.pt")

# Afficher les informations du modèle (facultatif)
model.info()

# Entraînez le modèle sur l'exemple de jeu de données COCO8 pour 100 époques
#results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640)

# Exécuter l'inférence avec le modèle YOLOv8n sur la vidéo
results = model("test2.mp4")

# Ouvrir la vidéo d'origine pour récupérer ses propriétés
cap = cv2.VideoCapture("test2.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Définir le writer vidéo pour enregistrer la sortie
output_path = "output_8.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec vidéo
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Parcourir les résultats et enregistrer la vidéo avec les annotations
for r in results:
    annotated_frame = r.plot()  # Générer une image annotée
    out.write(annotated_frame)  # Enregistrer la frame

    # Afficher chaque frame annotée (facultatif)
    """
    cv2.imshow("YOLO Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Quitter avec 'q'
        break
    """

# Libérer les ressources
cap.release()
out.release()
cv2.destroyAllWindows()

# Afficher les informations des objets détectés
for r in results:
    print(r.boxes)  # Coordonnées des boîtes englobantes
    print(r.names)  # Noms des classes détectées
    print(r.probs)  # Scores de confiance