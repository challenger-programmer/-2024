import os
import cv2
from ultralytics import YOLO
import numpy as np
from scipy.stats import beta
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# POSTリクエストを受け取ってデータを処理するエンドポイント
@app.route('/process', methods=['POST'])
def process_data():
    data = request.json  # JavaScriptから送信されたデータを受け取る
    
    # 受け取ったデータを表示
    print(f"Received data: {data}")
    
    # データを取得
    average_distance_syu = data['average_distance_syu']
    average_distance_ryu = data['average_distance_ryu']
    average_distance_deme = data['average_distance_deme']
    range_distance_syu = data['range_distance_syu']
    range_distance_ryu = data['range_distance_ryu']
    range_distance_deme = data['range_distance_deme']
    #hide_syubun_levelは小数で1～3 3が最も隠れている
    hide_syubun_level = data['hide_syubun_level']
    hide_ryukin_level = data['hide_ryukin_level']
    hide_demekin_level = data['hide_demekin_level']
    #集団率 0～1をとる 「0.65以上 3 仲いい」1~3
    fish_gather = data['fish_gather']

    # ここでPythonでデータ処理
    # 学習済みモデルのロード
    model = YOLO('C:/Users/tatsu/OneDrive/デスクトップ/高専プロコン/魚のAI学習2/runs/detect/train/weights/best.pt')  # ここに自分のトレーニング済みモデルを指定
    
    # 推論する画像が保存されているフォルダのパス
    image_folder = 'C:/Users/tatsu/OneDrive/デスクトップ/高専プロコン/魚のAI学習2/suiron/jpg'  # 推論対象のフォルダを指定
    output_folder = 'C:/Users/tatsu/OneDrive/デスクトップ/高専プロコン/魚のAI学習2/suiron/output'  # 出力先フォルダを指定
    os.makedirs(output_folder, exist_ok=True)  # 出力フォルダを作成
    
    # クラス名を設定 (syubunとryukinが含まれる前提)
    class_names = ['syubun', 'ryukin','demekin']
    
    
    #魚が物陰にいるときの画像の枚数
    hide_syubun = 0
    hide_ryukin = 0
    hide_demekin = 0
    
    
    
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
    
    
    #貰う-------
    # average_distance_syu = 0
    # average_distance_ryu = 0
    # average_distance_deme = 0
    # range_distance_syu = 0
    # range_distance_ryu = 0
    # range_distance_deme = 0

    # #集団率
    # fish_gather = [[0.5 for _ in range(len(class_names))] for _ in range(len(class_names))]
    #------
    
    #行動量一番活発4 1~4
    # activ_syubun = 1
    # activ_ryukin = 1
    # activ_demekin = 1

    #gather_probability = [[3 for _ in range(len(class_names))] for _ in range(len(class_names))]
    number = 0
    
    # フォルダ内の全画像に対して推論を行う
    for image_name in os.listdir(image_folder):
    
        number += 1
    
        # 拡張子が画像ファイル（.jpgや.png）でない場合はスキップ
        if not (image_name.endswith('.jpg') or image_name.endswith('.png')):
            continue
        
        image_path = os.path.join(image_folder, image_name)
    
        # 推論を実行
        results = model(image_path)
    
        # 推論結果を取得
        boxes = results[0].boxes  # 0番目の結果からバウンディングボックス情報を取得
    
        # syubunとryukinクラスに対するフィルタリング
        best_boxes = []
        # 画像を読み込む
        image = cv2.imread(image_path)
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
    
    

            
    
        
        # best_boxes のみを描画
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
    
    
                        #物陰の範囲cv2.rectangle(image, (380, 550), (570, 730), (0, 255, 0), 2)
                        if((box_center_x>=380 and box_center_x<=570 and box_center_y>=550 and box_center_y <= 730)or(box_center_x>=700 and box_center_x<=900 and box_center_y>=300 and box_center_y <= 660)):
                            hide_syubun += 1
                        
                    
                    elif model.names[int(box.cls.item())] == "ryukin":
                        # label += f" | state: {activ_ryukin}"  # 活動状態を追加
                        if(x_ryukin==0 and y_ryukin==0):
                            x_ryukin = box_center_x
                            y_ryukin = box_center_y
                        else:
                            #距離計算
                            movement_ryukin += ((box_center_x-x_ryukin)**2+(box_center_y-y_ryukin)**2)**0.5
    
                        #物陰cv2.rectangle(image, (700, 300), (900, 660), (0, 255, 0), 2)
    
                        if((box_center_x>=380 and box_center_x<=570 and box_center_y>=550 and box_center_y <= 730)or(box_center_x>=700 and box_center_x<=900 and box_center_y>=300 and box_center_y <= 660)):
                            hide_ryukin += 1
                        
    
                    elif model.names[int(box.cls.item())] == "demekin":
                        # label += f" | state: {activ_demekin}"  # 活動状態を追加
                        if(x_demekin==0 and y_demekin==0):
                            x_demekin = box_center_x
                            y_demekin = box_center_y
                        else:
                            #距離計算
                            movement_demekin += ((box_center_x-x_demekin)**2+(box_center_y-y_demekin)**2)**0.5
    
                        #物陰
                        if((box_center_x>=380 and box_center_x<=570 and box_center_y>=550 and box_center_y <= 730)or(box_center_x>=700 and box_center_x<=900 and box_center_y>=300 and box_center_y <= 660)):
                            hide_demekin += 1
                    
                    
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
                        
            
            
            
            # 画像を保存
            # output_path = os.path.join(output_folder, f'image_output_{image_name}')  # 元のファイル名を保持
            # cv2.imwrite(output_path, image)
        # print(fish_gather)
        # print(f"syubun:{hide_syubun} ryukin:{hide_ryukin} demekin:{hide_demekin}")
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
    #集団率 誰と仲の良いか
    # for i in range(len(class_names)):
    #         for j in range(i+1,len(class_names)):
    #             if(fish_gather[i][j]>0.65):
    #                 gather_probability[i][j]=3
    #             elif(fish_gather[i][j]>0.35):
    #                 gather_probability[i][j]=2
    #             else:
    #                 gather_probability[i][j]=1

    #隠居率
    if(hide_syubun>50):
        hide_syubun_level = 0.8 * hide_syubun_level + 0.2 * 3
    elif(hide_syubun>20):
        hide_syubun_level = 0.8 * hide_syubun_level + 0.2 * 2
    else:
        hide_syubun_level = 0.8 * hide_syubun_level + 0.2 * 1

    if(hide_ryukin>50):
        hide_ryukin_level = 0.8 * hide_ryukin_level + 0.2 * 3
    elif(hide_ryukin>20):
        hide_ryukin_level = 0.8 * hide_ryukin_level + 0.2 * 2
    else:
        hide_ryukin_level = 0.8 * hide_ryukin_level + 0.2 * 1

    if(hide_demekin>50):
        hide_demekin_level = 0.8 * hide_demekin_level + 0.2 * 3
    elif(hide_demekin>20):
        hide_demekin_level = 0.8 * hide_demekin_level + 0.2 * 2
    else:
        hide_demekin_level = 0.8 * hide_demekin_level + 0.2 * 1
     

    # 処理されたデータをクライアントに返す
    response_data = {
        "average_distance_syu": average_distance_syu,
        "average_distance_ryu": average_distance_ryu,
        "average_distance_deme": average_distance_deme,  # ここではそのまま返す
        "range_distance_syu":range_distance_syu,
        "range_distance_ryu":range_distance_ryu,
        "range_distance_deme":range_distance_deme,
        "hide_syubun_level":hide_syubun_level,
        "hide_ryukin_level":hide_ryukin_level,
        "hide_demekin_level":hide_demekin_level,
        "fish_gather":fish_gather,
        "activ_syubun":activ_syubun,
        "activ_ryukin":activ_ryukin,
        "activ_demekin":activ_demekin
    }
    
    return jsonify(response_data)  # 処理されたデータをJSONで返す

if __name__ == '__main__':
    app.run(debug=True)











    

    # for box in best_boxes:
    #     print(f"Class: {model.names[int(box.cls.item())]}, Confidence: {box.conf.item():.2f}, Coordinates: {box.xyxy}")  # .item()を使ってTensorから値を取得
    # print()  # 画像ごとに空行を入れて見やすくする
