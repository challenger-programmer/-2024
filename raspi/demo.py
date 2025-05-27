import cv2
import time
from ultralytics import YOLO
from scipy.stats import beta
import math

# カメラの初期化
cap = cv2.VideoCapture(0)

# フレーム取得間隔（秒）
interval = 0.5

#変数
average_distance_syu = 0
average_distance_ryu = 0
average_distance_deme = 0
range_distance_syu = 0
range_distance_ryu = 0
range_distance_deme = 0

#集団率 0～1をとる 「0.65以上 3 仲いい」1~3
fish_gather = [[0.5,0.5,0.5],[0.5,0.5,0.5],[0.5,0.5,0.5]]

#行動量 
movement_syubun = 0
movement_ryukin = 0
movement_demekin = 0
    
x_syubun = 0
y_syubun = 0
x_ryukin = 0
y_ryukin = 0
x_demekin = 0
y_demekin = 0

    # ここでPythonでデータ処理
    # 学習済みモデルのロード
model = YOLO('best.pt')


class_names = ['ryukin', 'syubun','demekin']

number = 0
display = "琉金:集団率[[s:],[d:]],移動量->活発度\n,朱文金:集団率[[r:],[d:]],移動量->活発度\n出目金:集団率[[r:],[s:]],移動量->活発度"
# 無限ループでデータを取得
while cap.isOpened():
    ret, image = cap.read()
    number+=1
    if not ret:
        print("カメラからフレームを取得できませんでした")
        break
    results = model(image)
    boxes = results[0].boxes 
    best_boxes = []

    img_height, img_width = image.shape[:2]  # 画像の幅と高さを取得
    img_center_x = img_width / 2  # 画像の中央のx座標を計算

    for class_name in class_names:
        # 該当クラスの検出結果を取得
        class_boxes = [box for box in boxes if model.names[int(box.cls.item())] == class_name]

        # スコアが最も高いものを選択
        if class_boxes:
            def distance_to_center(box):
                # バウンディングボックスの中心x座標を計算
                x1, y1, x2, y2 = box.xyxy.squeeze().tolist()
                box_center_x = (x1 + x2) / 2
                # 画像の中心との距離を返す
                return abs(box_center_x - img_center_x)

            # x座標が最も画像の中心に近いバウンディングボックスを選ぶ
            best_box = min(class_boxes, key=distance_to_center)
            best_boxes.append(best_box)

        if best_boxes:  # best_boxes が空でない場合
            fish_position = []
            fish_name =[]
    
            for i, box in enumerate(best_boxes, start=1):  # i を 1 から始める
                # バウンディングボックスの座標を取得
                if box.xyxy.numel() == 4:  # ちょうど4つの値があることを確認
                    x1, y1, x2, y2 = box.xyxy.squeeze().tolist()  # 必要に応じて squeeze を使用してフラット化
                    
                    # バウンディングボックスを描画
                    # cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # 線の色は緑
                    # クラス名と信頼度を描画
                    
                    fish_name.append(model.names[int(box.cls.item())])
                    # label = f"{model.names[int(box.cls.item())]}: {box.conf.item():.2f}"
                    box_center_x = (x1+x2)/2
                    box_center_y = (y1+y2)/2
                    fish_position.append([box_center_x,box_center_y])
    
    
    
    
                    if model.names[int(box.cls.item())] == "syubun":
                        # label += f" | state: {activ_syubun}"  # 活動状態を追加
                        if (x_syubun == 0 and  y_syubun==0):
                            x_syubun = box_center_x
                            y_syubun = box_center_y
                        else:
                            #距離計算
                            movement_syubun += ((box_center_x-x_syubun)**2+(box_center_y-y_syubun)**2)**0.5
    
                        
                    
                    elif model.names[int(box.cls.item())] == "ryukin":
                        # label += f" | state: {activ_ryukin}"  # 活動状態を追加
                        if(x_ryukin==0 and y_ryukin==0):
                            x_ryukin = box_center_x
                            y_ryukin = box_center_y
                        else:
                            #距離計算
                            movement_ryukin += ((box_center_x-x_ryukin)**2+(box_center_y-y_ryukin)**2)**0.5
    
                        
    
                    elif model.names[int(box.cls.item())] == "demekin":
                        # label += f" | state: {activ_demekin}"  # 活動状態を追加
                        if(x_demekin==0 and y_demekin==0):
                            x_demekin = box_center_x
                            y_demekin = box_center_y
                        else:
                            #距離計算
                            movement_demekin += ((box_center_x-x_demekin)**2+(box_center_y-y_demekin)**2)**0.5
                    
                    
                    # cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
            #集団率
            #平均を求める 200は奥行きの関係で引く
            
            for i,value in enumerate(fish_name,start=0):
                for j in range(len(class_names)):
                    if(value == class_names[j]):
                        fish_name[i]=j
                        break
                    
            if(len(fish_position)>1):
                for i in range(len(fish_position)):
                    for j in range(i+1,len(fish_position)):
                        #cv2.rectangle(image, (100, 180), (900, 760), (0, 255, 0), 2) 水槽
                        fish_position1 = fish_position[i]
                        fish_position2 = fish_position[j]
                        value_gather_x = abs(fish_position1[0]-fish_position2[0])/800
                        value_gather_y = abs(fish_position1[1]-fish_position2[1])/580
                        # ベータ分布のパラメータ
                        cumulative_probability_x = 1-beta.cdf(value_gather_x, 2, 4)
                        cumulative_probability_y = 1-beta.cdf(value_gather_y, 2, 4)
                        value_gather = (cumulative_probability_x+cumulative_probability_y)/2
                        fish_gather[max(fish_name[i],fish_name[j])][min(fish_name[i],fish_name[j])] = 0.8 * fish_gather[max(fish_name[i],fish_name[j])][min(fish_name[i],fish_name[j])] + 0.2 * value_gather
                        print(cumulative_probability_x)
    if number == 100:
        number = 0
        if(average_distance_syu-range_distance_syu>movement_syubun):
            activ_syubun = 1
        elif(average_distance_syu>movement_syubun):
            activ_syubun = 2
        elif(average_distance_syu+range_distance_syu>movement_syubun):
            activ_syubun = 3
        else:
            activ_syubun = 4
        if(average_distance_ryu-range_distance_ryu>movement_ryukin):
            activ_ryukin = 1
        elif(average_distance_ryu>movement_ryukin):
            activ_ryukin = 2
        elif(average_distance_ryu+range_distance_ryu>movement_ryukin):
            activ_ryukin = 3
        else:
            activ_ryukin = 4
        if(average_distance_deme-range_distance_deme>movement_demekin):
            activ_demekin = 1
        elif(average_distance_deme>movement_demekin):
            activ_demekin = 2
        elif(average_distance_deme+range_distance_deme>movement_demekin):
            activ_demekin = 3
        else:
            activ_demekin = 4
        range_distance_syu = 0.8 * range_distance_syu + 0.2 * abs(average_distance_syu - movement_syubun)/2
        range_distance_ryu = 0.8 * range_distance_ryu + 0.2 * abs(average_distance_ryu - movement_ryukin)/2
        range_distance_deme = 0.8 * range_distance_deme + 0.2 * abs(average_distance_deme - movement_demekin)/2
        average_distance_syu = 0.8 * average_distance_syu + 0.2 * movement_syubun 
        average_distance_ryu = 0.8 * average_distance_ryu + 0.2 * movement_ryukin 
        average_distance_deme = 0.8 * average_distance_deme + 0.2 * movement_demekin 
        movement_syubun = 0
        movement_ryukin = 0
        movement_demekin = 0
        x_syubun = 0
        y_syubun = 0
        x_ryukin = 0
        y_ryukin = 0
        x_demekin = 0
        y_demekin = 0
    display = f"琉金:集団率[[s:{math.ceilfish_gather[1][0]}],[d:{math.ceil(fish_gather[2][0])}]],移動量{movement_ryukin}->活発度{activ_ryukin}\n,朱文金:集団率[[r:{math.ceil(fish_gather[1][0])}],[d:{math.ceil(fish_gather[2][1])}]],移動量{movement_syubun}->活発度{activ_syubun}\n出目金:集団率[[r:{math.ceil(fish_gather[2][0])}],[s:{math.ceil(fish_gather[2][1])}]],移動量{movement_demekin}->活発度{activ_demekin}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (10, 30)  # テキストの位置（左上から10ピクセル, 30ピクセルの位置）
    font_scale = 1
    color = (0, 255, 0)  # 緑色のテキスト
    thickness = 2
    
    # テキストをフレームに描画
    cv2.putText(image, display, position, font, font_scale, color, thickness, cv2.LINE_AA)
    
    # フレームを表示（確認用）
    cv2.imshow("Camera Feed", image)
    
    # 'q' キーが押されたら終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # 0.5秒待機
    time.sleep(interval)

# リソース解放
cap.release()
cv2.destroyAllWindows()
