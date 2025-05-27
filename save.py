from flask import Flask, request, jsonify
import os
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# アップロードされた画像を保存するフォルダ
UPLOAD_FOLDER = r'C:\Users\tatsu\OneDrive\desktop\sourceprogram\website-backend\suiron\jpg'

# ファイルを受け取るエンドポイント
@app.route('/upload', methods=['POST'])
def upload_file():
    # リクエストにファイルが含まれているか確認
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']
    
    # ファイルが選択されているか確認
    if file.filename == '':
        return 'No selected file', 400

    if file:
        # ファイル名と保存先パスを決定
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        
        # ファイルを保存
        file.save(file_path)
        return f'File {file.filename} uploaded successfully!', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, ssl_context=('C:/Users/tatsu/OneDrive/desktop/certificate.crt', 'C:/Users/tatsu/OneDrive/desktop/private.key'))

