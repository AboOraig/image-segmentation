# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 11:25:18 2025

@author: maboorai279
"""

import os
import json

# Dossier contenant les JSON (annotations Cityscapes)
dossier_json = "cityscapes/labels/val_old"
# Dossier où enregistrer les fichiers YOLO .txt
dossier_yolo = "cityscapes/labels/val"

# Liste des classes à extraire (ex : tu peux adapter les noms selon tes besoins)
classes = ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'rider']

# Création du dossier de sortie s'il n'existe pas
os.makedirs(dossier_yolo, exist_ok=True)

def polygon_to_bbox(polygon):
    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]
    xmin = min(x_coords)
    xmax = max(x_coords)
    ymin = min(y_coords)
    ymax = max(y_coords)
    return xmin, ymin, xmax, ymax

def convert_annotation(json_path, output_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    image_width = data['imgWidth']
    image_height = data['imgHeight']

    lignes_yolo = []

    for obj in data['objects']:
        label = obj['label']

        if label not in classes:
            continue  # ignorer les classes non souhaitées

        class_id = classes.index(label)
        polygon = obj['polygon']
        xmin, ymin, xmax, ymax = polygon_to_bbox(polygon)

        # Coordonnées YOLO normalisées
        x_center = ((xmin + xmax) / 2) / image_width
        y_center = ((ymin + ymax) / 2) / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        ligne = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        lignes_yolo.append(ligne)

    # Nom du fichier texte (même que l'image)
    nom_fichier = os.path.splitext(os.path.basename(json_path))[0].replace("_gtFine_polygons", "") + ".txt"
    chemin_sortie = os.path.join(output_path, nom_fichier)

    with open(chemin_sortie, 'w') as f_out:
        f_out.write('\n'.join(lignes_yolo))

    print(f" Annotation convertie : {chemin_sortie}")

# Parcours des fichiers .json dans le dossier
for fichier in os.listdir(dossier_json):
    if fichier.endswith(".json"):
        chemin_json = os.path.join(dossier_json, fichier)
        convert_annotation(chemin_json, dossier_yolo)
