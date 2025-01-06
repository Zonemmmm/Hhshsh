import cv2 
import numpy as np

# Plages HSV initiales pour détecter le rouge
lower_red = np.array([0, 100, 100])  # Ajuster si nécessaire
#upper_red = np.array([359,60,62])
upper_red = np.array([10, 255, 255])

lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

# Initialisation de la capture vidéo
cap = cv2.VideoCapture(0)

def detect_fire(frame):
    """Détecte les flammes à partir des couleurs rouges dans l'image."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    return mask

def process_contours(mask, frame):
    """Dessine des rectangles autour des flammes détectées."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    flame_detected = False
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Filtrer les petites zones (ajuster si nécessaire)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Flamme detectee", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            flame_detected = True
    return flame_detected

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur lors de la capture d'image.")
        break

    # Détection des flammes
    mask = detect_fire(frame)

    # Traitement des contours pour la détection
    flame_detected = process_contours(mask, frame)

    # Afficher le masque pour ajuster les plages HSV
    cv2.imshow("Masque de Detection", mask)
    cv2.imshow("Detection de Flammes", frame)

    # Sauvegarder une image si une flamme est détectée
    if flame_detected and cv2.waitKey(1) & 0xFF == ord('r'):
        cv2.imwrite("flame_detected.jpg", frame)
        print("Image enregistrée : flame_detected.jpg")

    # Quitter si 'q' est pressé
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
