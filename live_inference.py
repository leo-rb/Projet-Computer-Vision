from ultralytics import YOLO
import cv2

# 1. Charger le modèle
model = YOLO('models/epi_model_v1.pt')

# 2. Ouvrir la webcam (0 est généralement la webcam par défaut)
cap = cv2.VideoCapture(0)

# Boucle infinie pour lire la vidéo image par image
while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        # Faire la prédiction sur l'image actuelle
        # conf=0.5 signifie qu'on n'affiche que si le modèle est sûr à 50%
        results = model(frame, conf=0.5)

        # Récupérer l'image annotée (avec les boîtes)
        annotated_frame = results[0].plot()

        # Afficher l'image dans une fenêtre
        cv2.imshow("Detection EPI - Appuyez sur 'q' pour quitter", annotated_frame)

        # Quitter si on appuie sur la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Libérer la caméra et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()