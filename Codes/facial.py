import numpy as np
import cv2
import face_recognition

imgAndrew = face_recognition.load_image_file('imagesBasic/Andrew Tate.jpg')
imgAndrew = cv2.cvtColor(imgAndrew, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('imagesBasic/Nicolas Cage.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgAndrew)[0]
encodeAndrew = face_recognition.face_encodings(imgAndrew)[0]
cv2.rectangle(imgAndrew, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeAndrew], encodeTest)
faceDis = face_recognition.face_distance([encodeAndrew], encodeTest)
print(results, faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 225), 2)

cv2.imshow('Andrew Tate', imgAndrew)
cv2.imshow('Andrew Tate test', imgTest)
cv2.waitKey(0)

