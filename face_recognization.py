import cv2

frontface_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_smile1.xml')

def detect(gray,frame):
    face=frontface_cascade.detectMultiScale(gray,1.3,5)
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x + w, y+h), (255, 0, 0), 1)
        cv2.putText(frame, "face_recognized", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        #cv2.circle(frame, (int((x+(w/2))),int((y+(h/2)))), int(w/2), (255, 0, 255), 2)
        gray_face = gray[y:y+h, x:x+w]
        frame_face = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(gray_face,1.1,22)
        for (ex, ey, ew, eh) in eyes:
            #cv2.rectangle(frame_face, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            cv2.circle(frame_face, (int((ex+(ew/2))),int((ey+(eh/2)))), int(ew/2), (0, 255, 0), 1)
            cv2.putText(frame_face, "eye", (ex, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        smile = smile_cascade.detectMultiScale(gray_face, 1.7, 22)
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(frame_face, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 1)
            cv2.putText(frame_face, "smile", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return frame

cam=cv2.VideoCapture(0)
while True:
    _, frame = cam.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('face detector', detect(gray, frame))
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
    
cam.release()
cv2.destroyAllWindows()