import cv2

# Önceden eğitilmiş nesne tanıma modellerini yükle
land_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "")
sea_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "")
jungle_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "")
desert_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "")

# Kamera başlatma
cap = cv2.VideoCapture(0)

while True:
    # Kameradan görüntü al
    ret, img = cap.read()

    # Gri tona dönüştür
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    land = land_cascade.detectMultiScale(gray, 1.3, 5)

    
    for (x, y, w, h) in land:
        # Dikdörtgen çiz ve etiketle
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    
    sea = sea_cascade.detectMultiScale(gray, 1.3, 5)

    
    for (ex, ey, ew, eh) in sea:
        # Dikdörtgen çiz ve etiketle
        cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        cv2.putText(img, 'Eye', (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    
    jungle = jungle_cascade.detectMultiScale(gray, 1.1, 3)


    for (x, y, w, h) in jungle:
        # Dikdörtgen çiz ve etiketle
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(img, 'Car', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    
    dezert = desert_cascade.detectMultiScale(gray, 1.1, 3)

    
    for (x, y, w, h) in dezert:
        # Dikdörtgen çiz ve etiketle
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.putText(img, 'Pedestrian', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Görüntüyü göster
    cv2.imshow('Biome Detection', img)

    # Çıkış için 'q' tuşuna basıldığını kontrol et
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı kapat
cap.release()
cv2.destroyAllWindows()
