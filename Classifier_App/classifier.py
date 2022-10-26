import os, shutil
from flask import Flask, request, redirect, url_for, render_template, Markup
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from PIL import Image
import numpy as np
import cv2
from face_rectangle import get_face_rect

UPLOAD_FOLDER = "./static/images/"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

labels = ["齋藤飛鳥","玉森裕太"]
# n_class = len(labels)
# img_size = 32
# n_result = 1  # 上位3つの結果を表示

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/result", methods=["GET","POST"])
def result():
    if request.method == "POST":
        # ファイルの存在と形式を確認
        if "file" not in request.files:
            print("File doesn't exist!")
            return redirect(url_for("index"))
        file = request.files["file"]
        if not allowed_file(file.filename):
            print(file.filename + ": File not allowed!")
            return redirect(url_for("index"))

        # ファイルの保存
        if os.path.isdir(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
        os.mkdir(UPLOAD_FOLDER)
        filename = secure_filename(file.filename)  # ファイル名を安全なものに
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        #元の画像
        test_img=cv2.imread(filepath)
        test_img=cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        #元画像から顔部分切り取った画像
        face_img, d=get_face_rect(filepath)
        face_img=cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        # plt.imshow(face_img)
        # plt.axis('off')
        # plt.show()

        #データの型変換
        x_test=np.asarray(face_img)
        x_test=x_test.reshape(-1, 64, 64, 3)

        #元画像に顔部分の矩形を描画
        x1=int(d.left())
        y1=int(d.top())
        x2=int(d.right())
        y2=int(d.bottom())
        cv2.rectangle(test_img, (x1, y1), (x2, y2), color=(255,0,0), thickness=5)
        # plt.imshow(test_img)
        # plt.axis('off')
        # plt.show()

        #モデルでの予測
        model = load_model("./images/model_asutama.h5")
        pred=model.predict(x_test)[0]
        print(pred[0])
        if pred[0]<0.5:
            # print("予測結果：", labels[0])
            result=labels[0]
        else:
            # print("予測結果：", labels[1])
            result=labels[1]

    return render_template("result.html", result=Markup(result), filepath=filepath)

    #     # 画像の読み込み
    #     image = Image.open(filepath)
    #     image = image.convert("RGB")
    #     image = image.resize((img_size, img_size))
    #     x = np.array(image, dtype=float)
    #     x = x.reshape(1, img_size, img_size, 3) / 255
    #
    #     # 予測
    #     model = load_model("./images/model_asutama.h5")
    #     y = model.predict(x)[0]
    #     sorted_idx = np.argsort(y)[::-1]  # 降順でソート
    #     result = ""
    #     for i in range(n_result):
    #         idx = sorted_idx[i]
    #         ratio = y[idx]
    #         label = labels[idx]
    #         result += "<p>" + str(round(ratio*100, 1)) + "%の確率で" + label + "です。</p>"
    #     return render_template("result.html", result=Markup(result), filepath=filepath)
    # else:
    #     return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
