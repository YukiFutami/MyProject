import cv2

def test_camera():
    cap = cv2.VideoCapture(0)  # カメラのインデックスを指定
    if not cap.isOpened():
        print("カメラが接続されていません")
        return
    print("カメラが接続されています。画像を表示します。終了するには 'q' を押してください。")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("画像を取得できませんでした")
            break
        cv2.imshow('Camera Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

test_camera()

