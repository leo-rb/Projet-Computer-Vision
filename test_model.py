from ultralytics import YOLO
import os
import random

# 1. Charger ton modèle entrainé
# Assure-toi que le chemin est bon !
model = YOLO('models/epi_model_v1.pt') 

# 2. Choisir une image au hasard dans le dossier de validation
valid_images_path = os.path.join('notebooks', 'PPE-detection-1', 'valid', 'images')
random_image = random.choice(os.listdir(valid_images_path))
image_path = os.path.join(valid_images_path, random_image)

print(f"Test sur l'image : {image_path}")

# 3. Faire la prédiction
results = model(image_path, show=True) # show=True ouvre une fenêtre avec le résultat

# 4. (Optionnel) Sauvegarder l'image résultat pour la voir si la fenêtre ne s'ouvre pas
results[0].save(filename='resultat_test.jpg')
print("Image sauvegardée sous 'resultat_test.jpg'")