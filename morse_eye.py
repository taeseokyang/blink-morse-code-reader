import cv2  
import dlib 
import imutils 
from scipy.spatial import distance as dist 
from imutils import face_utils 
import time
  

def calculate_EAR(eye):
    y1 = dist.euclidean(eye[1], eye[5])
    y2 = dist.euclidean(eye[2], eye[4])
    x1 = dist.euclidean(eye[0], eye[3])
    EAR = (y1+y2) / x1
    return EAR

def calculate_MAR(left,right,top,bottom):
    y1 = dist.euclidean(top, bottom)
    x1 = dist.euclidean(left, right)
    MAR = y1 / x1
    return MAR

morse_code_decode = {
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

# 시간 상수 설정
BLINK_THRESH = 0.35
MOUSE_OPEN_THRESH = 0.55
REMOVE_WORD_TIME = 2
END_CODE_TIME = 1.5
BLINK_LONG_TIME = 0.3

close_eye_before = False
open_mouse_before = False
end_code_writing = True
write_mode_on = True

word = ""
morse_code = ""

# 시간 체크
close_eye_time = 0
open_eye_time = 0
open_eye_time_for_end_code = 0
close_eye_time_for_end_code = 0 
blink_time=0
close_mouse_time = 0
close_mouse_time = 0


(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
mouse_top_id = 63
mouse_bottom_id = 67
mouse_left_id = 49
mouse_right_id = 55
  
cam = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
landmark_predict = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
while 1:
        ret, frame = cam.read()
        frame = imutils.resize(frame, width=640)

        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
        faces = detector(img_gray)
        
        for face in faces:
            # 랜드마크 탐지
            shape = landmark_predict(img_gray, face)
            shape = face_utils.shape_to_np(shape)

            # 랜드마크 추출
            lefteye = shape[L_start: L_end]
            righteye = shape[R_start:R_end]
            mouse_top = shape[mouse_top_id]
            mouse_bottom = shape[mouse_bottom_id]
            mouse_left = shape[mouse_left_id]
            mouse_right = shape[mouse_right_id]
            
            # 눈비, 입비 계산
            left_EAR = calculate_EAR(lefteye)
            right_EAR = calculate_EAR(righteye)
            mouse_MAR = calculate_MAR(mouse_left,mouse_right,mouse_top,mouse_bottom)

            avg = (left_EAR+right_EAR)/2

            x = face.left()
            y = face.top() #could be face.bottom() - not sure
            w = face.right() - face.left()
            h = face.bottom() - face.top()

            # 탐지 영역 그리기
            cv2.rectangle(frame, (x, y), (x+w, y+h),(0, 255, 0), 2)
            cv2.putText(frame, 'me',  (x, y-20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 200, 0), 1)
            for (x, y) in shape[L_start: L_end]:cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            for (x, y) in shape[R_start: R_end]:cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            
            #입열림 판단
            if mouse_MAR >= MOUSE_OPEN_THRESH:
                open_mouse_before = True
                open_mouse_time2 = time.time()

                # 글자 삭제, 전체 삭제
                if write_mode_on == False:
                    if open_mouse_time2 - close_mouse_time <= REMOVE_WORD_TIME:
                        cv2.putText(frame, str(REMOVE_WORD_TIME-(open_mouse_time2 - close_mouse_time))[:3]+"s left for remove all", (120, 300), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
                    else:
                        word = ""
                # 글자 삭제, 한 글자 삭제
                else:
                    if open_mouse_time2 - close_mouse_time <= REMOVE_WORD_TIME:
                        cv2.putText(frame, str(REMOVE_WORD_TIME-(open_mouse_time2 - close_mouse_time))[:3]+"s left for remove one", (120, 300), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
                    else:
                        word = word[:-1]#수정해야함.
                        close_mouse_time = time.time()
            else:
                close_mouse_time = time.time()
                open_mouse_before = False
                
            # 눈깜빡임 인식    
            if avg <= BLINK_THRESH:
                if close_eye_before == False:
                    close_eye_time = time.time()# 감았을 때 시간
                close_eye_before = True
                end_code_writing=False
            else:
                if close_eye_before == True:
                    open_eye_time = time.time()# 떳을 때 시간
                    open_eye_time_for_end_code = time.time()# 모스부호 확정을 위한 시간 시작
                    blink_time = open_eye_time - close_eye_time
                    if blink_time <= BLINK_LONG_TIME:# 모스부호 판단
                        morse_code+='.'
                    else:
                        morse_code+="_"
                else:
                    close_eye_time_for_end_code = time.time() # 모스부호 확정을 위한 시간 끝
                    if close_eye_time_for_end_code - open_eye_time_for_end_code <= END_CODE_TIME:
                        cv2.putText(frame, str(END_CODE_TIME-(close_eye_time_for_end_code - open_eye_time_for_end_code))[:3]+"s left for selct", (200, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)

                    # 출력
                    if close_eye_time_for_end_code - open_eye_time_for_end_code > END_CODE_TIME and end_code_writing == False:
                        if morse_code in morse_code_decode:
                            print(morse_code_decode[morse_code])
                            word+=morse_code_decode[morse_code]
                        morse_code = ""
                        end_code_writing = True
                close_eye_before = False
                    
            # 정보
            cv2.putText(frame, morse_code, (shape[34][0]-(len(morse_code)*5+10),shape[34][1]+5), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
            cv2.putText(frame, "EAR:"+str(avg)[:4], (20, 70), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 200, 0), 1)
            cv2.putText(frame, "MAR:"+str(mouse_MAR)[:4], (20, 90), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 200, 0), 1)
            cv2.putText(frame,  "blink_time:"+str(blink_time)[:4], (20, 110), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 200, 0), 1)
            cv2.putText(frame, word,(2, 350), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
            if morse_code in morse_code_decode:
                cv2.putText(frame, morse_code_decode[morse_code],  shape[34]-20 , cv2.FONT_HERSHEY_DUPLEX, 1, (200, 0, 0), 1)
            break

        
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  
cam.release()
cv2.destroyAllWindows()