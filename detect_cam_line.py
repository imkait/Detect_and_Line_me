import cv2
import time
from ultralytics import YOLO
import requests

# 發送訊息時間，放在迴圈前
stime=0 
# i:連續推論值變數
i=-1
# c:連續推論記數
c=0

# 發送line的副程式
def lineme():
    url = 'https://notify-api.line.me/api/notify'
    token = '你的line權杖'
    headers = {'Authorization': 'Bearer ' + token}
    data = {'message':'偵測異常通知'}
    image=open('line.jpg', 'rb')
    imageFile={'imageFile': image}
    r=requests.post(url, headers=headers, data=data, files=imageFile)
    print(r.status_code)
# 發送line訊息副程式結束


# 載入Yolov8模型
model = YOLO("yolov8n.pt",task='Detect')

# 設定影像來源，若只有一顆WebCam設為0，也可以是檔案
video_src = 0
cap = cv2.VideoCapture(video_src)
#設定畫面寬高
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 當鏡頭開啟
while cap.isOpened():
    # 讀取畫面
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    #若成功讀取畫面
    if success:
        # 將畫面送到模型推論，並回傳結果
        # predict參數：https://docs.ultralytics.com/modes/predict/#inference-sources
        # result的屬性與方法：https://docs.ultralytics.com/modes/predict/#working-with-results
        results = model.predict(frame,classes=[0])
        
        #將結果畫面回傳
        #plot方法參數：https://docs.ultralytics.com/modes/predict/#plot-method-parameters
        img= results[0].plot()
        

        '''發送line訊息主程式開始'''
        # 將畫面中所有偵測到的類別id存入串列
        mydetect=[int(x) for x in results[0].boxes.cls]

        # 如果i是未指定值或已經是分類值
        if i==-1 or i in mydetect:
            i=0
            c+=1
        else:
            # i並非未指定值,亦非目前分類值，則從新計算
            i=-1
            c=0
        
        if c>=10:
            # 目前時間與上一次發送的時間超過10s
            if int(time.time()) -stime > 10:
                #擷取畫面
                cv2.imwrite('line.jpg',img)
                #發送一次訊息
                lineme()
                #計數重新計算
                c=0
                #重新設定已發送時間
                stime=int(time.time())

        '''發送line訊息主程式結束'''

        #呈現畫面
        cv2.imshow("YOLOv8-Detect",img)

        #按q鍵離開
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 如果無影像則中止
        break

# 釋放鏡頭及關閉視窗
cv2.destroyAllWindows()
