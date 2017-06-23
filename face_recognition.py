
import face
import cv2
import sys
import time

ff = face.FACE()

videofile = sys.argv[1]
cap = cv2.VideoCapture(videofile)

count =0

while(True):
    s_time = time.time()
    ret, frame = cap.read()
    frame = ff.classification(frame)
    cv2.imshow('', frame)
    
    print ('####### 1 frame : {}'.format(time.time()-s_time))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
