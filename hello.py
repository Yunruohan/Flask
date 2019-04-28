from flask import Flask
from flask import request
import cv2
import urllib
import numpy as np
from hogGet import Hog_descriptor
from keras.models import load_model
import json
import keras
import datetime
import pymysql
import matplotlib.pyplot as plt # plt
import os
app = Flask(__name__)
db = pymysql.connect(host="120.79.17.239", user="root", password="111111", db="sdm", port=3306)
cursor = db.cursor()
cursor.execute("select * from picture_record")
data = cursor.fetchone()
print(data)

@app.route('/hogGet', methods=['GET'])
def HogGet():
    # imgurl = request.args.get('url')
    params = request.args
    imgurl = params.get('url')
    userid = params.get('userid')
    resp = urllib.request.urlopen(imgurl)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    img = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    hog = Hog_descriptor(img, cell_size=6, bin_size=6)
    vector, image = hog.extract()
    plt.imsave('http://120.79.17.239', image)
    dir_name = os.path.join("/home/imgs", )
    return str()

@app.route('/predict', methods=['GET'])
def predict():
    keras.backend.clear_session()
    params = request.args
    imgurl = params.get('url')
    userid = params.get('userid')
    resp = urllib.request.urlopen(imgurl)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    img = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    hog = Hog_descriptor(img, cell_size=6, bin_size=6)
    vector, image = hog.extract()
    X = np.array(vector)
    model = load_model('model.h5')
    result = model.predict(X)
    success = result[0].tolist()
    sql = "INSERT INTO "
    return str(success)

def success(data, resperr='', debug=False, escape=True, encoder=None):
    ret = {"respcd": "0000", "resperr": resperr, "respmsg": "", "data": data}
    return json.dumps(ret, ensure_ascii=escape, cls=encoder, separators=(',', ':'), default = json_default_trans)

def json_default_trans(obj):
    '''json对处理不了的格式的处理方法'''
    if isinstance(obj, datetime.datetime):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    if isinstance(obj, datetime.date):
        return obj.strftime('%Y-%m-%d')
    raise TypeError('%r is not JSON serializable' % obj)

if __name__ == '__main__':
    app.run(debug=True, host='172.100.108.192', port=8088)
    # app.run(debug=True, host='127.0.0.1', port=8088)
    # url = 'http://120.79.17.239/1009/25f50ec6-5c28-11e9-8a77-00163e021e9c.png'
    # 发现下面的代码只适合读取jpg图 不适合读取png图
    # url = 'http://wx2.sinaimg.cn/mw690/ac38503ely1fesz8m0ov6j20qo140dix.jpg'
    # cap = cv2.VideoCapture(url)
    # print(cap.isOpened())
    # if (cap.isOpened()):
    #     ret, img = cap.read()
    #     print(img)
    #     print(ret)
    #     cv2.imshow("image", img)
    #     cv2.waitKey()
    # 方法二 利用urllib实现
    # resp = urllib.request.urlopen(url)
    # image = np.asarray(bytearray(resp.read()), dtype="uint8")
    # img = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

    # 接口一 hog提取改为接口模式一
    # hog = Hog_descriptor(img, cell_size=6, bin_size=6)
    # vector, image = hog.extract()
    # base64_data = base64.b64encode(image)
    # print(str(base64_data))

    # 接口二 hog提取+模型预测改为接口
    # hog = Hog_descriptor(img, cell_size=6, bin_size=6)
    # vector, image = hog.extract()
    # print(vector[0])
    # X = np.array(vector)
    # model = load_model('model.h5')
    # result = model.predict(X)
    # print(result[0].tolist())

    # 接口三 尝试先将读取到的图片计算为model可接受的csv形式 目前可猜测的计算是当前的像素值除以255
    # img2 = [[img[i][j] /255 for j in range(len(img[i]))] for i in range(len(img))]
    # test = np.array(img2)
    # test = test.reshape(80, 80, 1)
    # model = load_model('mode_Img2Contour/model_1.h5')
    # result = model.predict(test)
    # print(result)

    # 接口四 尝试使用学长的Img对应的model来预测
    # im  = np.loadtxt(r'E:\Python\data\images_2900_csv\IMG_1.csv', delimiter=',')
    # images = np.zeros((1, 80, 80))
    # images[0, :, :] = im
    # images = images.astype('float32')
    # model = load_model('mode_Img2Contour/model_1.h5')
    # res = model.predict(images)
    # print(res)
