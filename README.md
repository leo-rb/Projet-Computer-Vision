# 🏗️ Système de Détection d'EPI (Équipements de Protection Individuelle)

Ce projet utilise le Deep Learning (YOLOv8) pour détecter en temps réel si les ouvriers sur un chantier portent leurs équipements de sécurité (Casques, Gilets, etc.).

## Objectif
Améliorer la sécurité sur les chantiers en automatisant la surveillance du port des EPI via des caméras de surveillance.

## Tech Stack
* **Langage :** Python 3.11
* **Modèle :** YOLOv8 (Fine-tuné)
* **Librairies :** Ultralytics, OpenCV, Roboflow

## Comment lancer le projet
1. Cloner le repo :
   ```bash
   git clone [https://github.com/leo-rb/Projet-Computer-Vision.git](https://github.com/leo-rb/Projet-Computer-Vision.git)


2. Dépendances
```bash
pip install -r requirements.txt
```

3. Démo webcam 
Préparer une photo d'une personne (sur son téléphone perso) équipée d'équipements de sécurité

```bash
python live_inference.py
