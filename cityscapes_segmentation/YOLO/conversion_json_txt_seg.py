# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 10:16:49 2025

@author: maboorai279

Ce script convertit les annotations de segmentation au format JSON provenant du jeu de données Cityscapes
en un format compatible avec YOLO pour la segmentation d'objets.
Les annotations sont normalisées et filtrées en fonction d'une liste de classes spécifiques,
avant d'être enregistrées dans un fichier texte pour chaque image.
Les annotations incluent les coordonnées normalisées des polygones représentant les objets détectés.
"""

import os
import json

# === CONFIGURATION ===
# Répertoire des fichiers JSON (annotations originales)
dossier_json = "cityscapes0/labels/val_old"

# Répertoire de sortie des annotations YOLO segmentation
dossier_labels = "cityscapes0/labels/val_new"

# Liste des classes que tu veux garder (dans cet ordre)
classes = ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'rider']

# Créer le dossier de sortie s’il n’existe pas
os.makedirs(dossier_labels, exist_ok=True)

def normaliser(point, largeur, hauteur):
    """Normalise un point (x, y) entre 0 et 1"""
    x, y = point
    return x / largeur, y / hauteur

def convertir_fichier(json_path, dossier_sortie):
    with open(json_path, "r") as f:
        data = json.load(f)

    largeur = data["imgWidth"]
    hauteur = data["imgHeight"]
    lignes = []

    for obj in data["objects"]:
        label = obj["label"]
        if label not in classes:
            continue

        id_classe = classes.index(label)
        polygone = obj["polygon"]

        points_normalises = []
        for point in polygone:
            x, y = normaliser(point, largeur, hauteur)
            points_normalises.extend([f"{x:.6f}", f"{y:.6f}"])

        ligne = f"{id_classe} " + " ".join(points_normalises)
        lignes.append(ligne)

    if lignes:
        nom_image = os.path.basename(json_path).replace("_gtFine_polygons.json", "")
        sortie_path = os.path.join(dossier_sortie, f"{nom_image}.txt")
        with open(sortie_path, "w") as f_out:
            f_out.write("\n".join(lignes))
        print(f" Converti : {sortie_path}")
    else:
        print(f" Ignoré (aucun objet utile) : {json_path}")

# === Exécution sur tous les fichiers JSON du dossier ===
for fichier in os.listdir(dossier_json):
    if fichier.endswith(".json"):
        chemin_json = os.path.join(dossier_json, fichier)
        convertir_fichier(chemin_json, dossier_labels)