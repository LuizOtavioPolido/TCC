import cv2
import pytesseract

coordX = 0
coordY = 0
coordW = 0
coordH = 0


# Carregue o classificador em cascata para placas de carro
placa_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# Carregue a imagem onde deseja detectar a placa
imagem = cv2.imread('carro7.jpg')

# Converta a imagem para escala de cinza, pois o classificador funciona melhor em imagens em preto e branco
gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Detectar placas na imagem
placas = placa_cascade.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Desenhar retângulos ao redor das placas detectadas
for (x, y, w, h) in placas:
    cv2.rectangle(imagem, (x, y), (x+w, y+h), (255, 0, 0), 2)
    coordX = x
    coordY = y
    coordW = w
    coordH = h

# Recortar a imagem original na área da placa detectada
if coordW > 0 and coordH > 0:  # Verificar se há alguma detecção
    imagem_recortada = imagem[coordY:coordY+coordH, coordX:coordX+coordW]

    # Mostrar a imagem original com as detecções
    cv2.imshow('Placas detectadas', imagem)

    # Mostrar a imagem recortada
    cv2.imshow('Imagem Recortada', imagem_recortada)

    # Esperar até que uma tecla seja pressionada e fechar as janelas
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Nenhuma placa detectada.")
