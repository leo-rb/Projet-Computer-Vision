from ultralytics import YOLO
import os
import random

# Chargement du modèle entrainé
model = YOLO('models/epi_model_v1.pt') 

# 2. Choix d'image random
valid_images_path = os.path.join('notebooks', 'PPE-detection-1', 'valid', 'images')
random_image = random.choice(os.listdir(valid_images_path))
image_path = os.path.join(valid_images_path, random_image)

print(f"Test sur l'image : {image_path}")

# 3. Faire la prédiction
results = model(image_path, show=True) # show=True ouvre une fenêtre avec le résultat

results[0].save(filename='resultat_test.jpg')
print("Image sauvegardée sous 'resultat_test.jpg'")