#https://www.geeksforgeeks.org/eye-blink-detection-with-opencv-python-and-dlib/
# Importing the required dependencies
import cv2  # for video rendering
import dlib  # for face and landmark detection
import imutils
# for calculating dist b/w the eye landmarks
from scipy.spatial import distance as dist
# to get the landmark ids of the left and right eyes
# you can do this manually too
from imutils import face_utils
import time
  
# from imutils import
  
cam = cv2.VideoCapture(0)
  
# defining a function to calculate the EAR
def calculate_EAR(eye):
  
    # calculate the vertical distances
    y1 = dist.euclidean(eye[1], eye[5])
    y2 = dist.euclidean(eye[2], eye[4])
  
    # calculate the horizontal distance
    x1 = dist.euclidean(eye[0], eye[3])
  
    # calculate the EAR
    EAR = (y1+y2) / x1
    return EAR


def calculate_MAR(left,right,top,bottom):
  
    # calculate the vertical distances
    y1 = dist.euclidean(top, bottom)
  
    # calculate the horizontal distance
    x1 = dist.euclidean(left, right)
  
    # calculate the EAR
    MAR = y1 / x1
    return MAR
# Variables
morse_code = {
    '._':'A',
    '_...':'B',
    '_._.':'C',
    '_..':'D',
    '.':'E',
    '.._.':'F',
    '__.':'G',
    '....':'H',
    '..':'I',
    '.___':'J',
    '_._':'K',
    '._..':'L',
    '__':'M',
    '_.':'N',
    '___':'O',
    '.__.':'P',
    '__._':'Q',
    '._.':'R',
    '...':'S',
    '_':'T',
    '.._':'U',
    '..._':'V',
    '.__':'W',
    '_.._':'X',
    '_.__':'Y',
    '__..':'Z',
    '_____':'0',
    '.____':'1',
    '..___':'2',
    '...__':'3',
    '...._':'4',
    '.....':'5',
    '_....':'6',
    '__...':'7',
    '___..':'8',
    '____.':'9'
    }
word = ""
blink_thresh = 0.35
count_frame = 0
check_eye = 0
check_mouse = 0
remove_time = 2
end_code = 1
end_code_time = 1.5
open_thresh = 0.55
code = ""
start = 0
end = 0
start2 = 0
long=0
end2 = 0
mouse_start = 0
can_write = 0
# Eye landmarks
(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
mouse_top_id = 63
mouse_bottom_id = 67
mouse_left_id = 49
mouse_right_id = 55
  
# Initializing the Models for Landmark and 
# face Detection
detector = dlib.get_frontal_face_detector()
landmark_predict = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
while 1:
  

        ret, frame = cam.read()
        frame = imutils.resize(frame, width=640)
  
        # converting frame to gray scale to
        # pass to detector
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
        # detecting the faces
        faces = detector(img_gray)
        
        for face in faces:
            # landmark detection
            shape = landmark_predict(img_gray, face)
  
            # converting the shape class directly
            # to a list of (x,y) coordinates
            shape = face_utils.shape_to_np(shape)
  
            # parsing the landmarks list to extract
            # lefteye and righteye landmarks--#
            lefteye = shape[L_start: L_end]
            righteye = shape[R_start:R_end]
            mouse_top = shape[mouse_top_id]
            mouse_bottom = shape[mouse_bottom_id]
            mouse_left = shape[mouse_left_id]
            mouse_right = shape[mouse_right_id]
            
            # cv2.rectangle(frame, (lefteye[0],lefteye[0]), (lefteye[0]+lefteye[3], lefteye[0]+lefteye[3]),(0, 255, 0), 2)
  
            # Calculate the EAR
            left_EAR = calculate_EAR(lefteye)
            right_EAR = calculate_EAR(righteye)

            mouse_MAR = calculate_MAR(mouse_left,mouse_right,mouse_top,mouse_bottom)
            # Avg of left and right eye EAR
            avg = (left_EAR+right_EAR)/2
            x = face.left()
            y = face.top() #could be face.bottom() - not sure
            w = face.right() - face.left()
            h = face.bottom() - face.top()
            cv2.rectangle(frame, (x, y), (x+w, y+h),(0, 255, 0), 2)
            cv2.putText(frame, 'me',  (x, y-20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 200, 0), 1)
            for (x, y) in shape[L_start: L_end]:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            for (x, y) in shape[R_start: R_end]:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            

            if mouse_MAR >= open_thresh:
                check_mouse = 1
                mouse_end2 = time.time()
                if can_write == 0:
                    if mouse_end2 - mouse_start <= remove_time:
                        cv2.putText(frame, str(remove_time-(mouse_end2 - mouse_start))[:3]+"s left for remove all", (120, 300), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
                    else:
                        word = ""
                else:
                    if mouse_end2 - mouse_start <= remove_time:
                        cv2.putText(frame, str(remove_time-(mouse_end2 - mouse_start))[:3]+"s left for remove one", (120, 300), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
                    else:
                        word = word[:-1]
                        mouse_start = time.time()
            else:
                if check_mouse == 1:
                    mouse_end = time.time()
                    if can_write == 0:
                        if mouse_end - mouse_start <= remove_time:
                            can_write = 1
                            code = ""
                    else:
                        if mouse_end - mouse_start <= remove_time:
                            can_write = 0
                            code = ""
                mouse_start = time.time()
                check_mouse = 0
                

            
     
            if can_write:
                cv2.putText(frame, 'Detecting', (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (200, 0, 0), 1)
                if avg < blink_thresh:
                    if check_eye == 0:
                        start = time.time()
                    check_eye = 1
                    end_code=0
                else:

                    if check_eye == 1:
                        count_frame += 1  # incrementing the frame count
                        end = time.time()
                        start2 = time.time()
                        long = end - start
                        if long <= 0.3:
                            code+='.'
                        else:
                            code+="_"
                    else:
                        end2 = time.time()
                        if end2 - start2 <= end_code_time:
                            cv2.putText(frame, str(end_code_time-(end2 - start2))[:3]+"s left for selct", (200, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)


                        if end2 - start2 > end_code_time and end_code == 0:
                            if code in morse_code:
                                print(morse_code[code])
                                word+=morse_code[code]
                            code = ""
                            end_code = 1
                    check_eye = 0
            else:
                cv2.putText(frame, 'Open mouse quickly : start write code', (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 200, 0), 1)
                cv2.putText(frame, 'Open mouse as long : remove word', (20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 200, 0), 1)
                    
                    
            cv2.putText(frame, code, (shape[34][0]-(len(code)*5+10),shape[34][1]+5), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
            cv2.putText(frame, "EAR:"+str(avg)[:4], (20, 70), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 200, 0), 1)
            cv2.putText(frame, "MAR:"+str(mouse_MAR)[:4], (20, 90), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 200, 0), 1)
            cv2.putText(frame, "Long:"+str(long)[:4], (20, 110), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 200, 0), 1)
            cv2.putText(frame, word,(2, 350), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
            if code in morse_code:
                cv2.putText(frame, morse_code[code],  shape[34]-20 , cv2.FONT_HERSHEY_DUPLEX, 1, (200, 0, 0), 1)
            break

            
        
  
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  
cam.release()
cv2.destroyAllWindows()