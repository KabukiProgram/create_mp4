import sys
import cv2
import os

# カメラの読込み
# 内蔵カメラがある場合、下記引数の数字を変更する必要あり
cap = cv2.VideoCapture(0)

#カスケード分類器読み込み
haarascades_path = os.getcwd() + "/haarcascades/"
face_cascade = cv2.CascadeClassifier(haarascades_path+"haarcascade_frontalface_alt.xml")
eye_cascade = cv2.CascadeClassifier(haarascades_path+'haarcascade_eye_tree_eyeglasses.xml')


if not cap.isOpened():
    print("カメラが正常ではありません")
    exit()

#動画生成
filePath = './copy.mp4'
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) # フレームレート

print('fps : '+str(fps))

codec = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(filePath, codec, fps, (W, H))


# 動画終了まで、1フレームずつ読み込んで表示する。
while(cap.isOpened()):
    # 1フレーム毎　読込み
    ret, frame = cap.read()
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    lists_face = face_cascade.detectMultiScale(frame_gray, minSize=(100, 100))

    lists_eye = eye_cascade.detectMultiScale(frame_gray, minSize=(50, 50))

    for (x,y,w,h) in lists_face:
        img = cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), thickness=2)
        video.write(img)

    for (x,y,w,h) in lists_eye:
        img = cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), thickness=2)
        video.write(img)
    
    # カメラから読込んだ映像をファイルに書き込む
    video.write(frame)

    # GUIに表示
    cv2.imshow("Camera", frame)
    #cv2.imshow('video image', frame)

    # qキーが押されたら途中終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 終了処理
video.release()
cap.release()
cv2.destroyAllWindows()
