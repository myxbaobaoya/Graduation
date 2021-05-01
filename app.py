import train_model
import UseUpdate
import os
import time
from flask import Flask, request, jsonify,send_file
from flask_cors import CORS

app = Flask(__name__)

CORS(app, resources={r"/app": {"origins": "*"}})


@app.route('/app',methods=['GET','POST'])
def getData():
    if request.method== 'POST':
        print(request.form)
        if request.form['type'] == 'train':
            file_train = request.files['filetrain']
            filepath_train='train_data.txt'              #按照train和result的字母顺序，先传过来的是result
            file_train.save(filepath_train)
            file_train = request.files['fileresult']
            filepath_result = 'result_data.txt'
            file_train.save(filepath_result)
            var1,var2,var3=NewModel()
            time.sleep(40)
            if os.path.exists('model1.pkl') and os.path.exists('model2.pkl'):
                print('exist')
                return jsonify({'type': 'modelfinish', 'var1': var1, 'var2': var2, 'var3': var3})
        elif request.form['type'] == 'predict':
            file_train = request.files['filepre']
            filepath_train = 'predict.txt'
            file_train.save(filepath_train)
            Predict()
            time.sleep(20)
            if os.path.exists('predict_result1.txt') and os.path.exists('predict_result2.txt')  and os.path.exists('predict_result3.txt'):
                print('exist')
                return 'resultfinish'
        elif request.form['type'] == 'update':
            file_train = request.files['updatefiletrain']
            filepath_train = 'update_train.txt'  # 按照train和result的字母顺序，先传过来的是result
            file_train.save(filepath_train)
            file_train = request.files['updatefileresult']
            filepath_result = 'update_result.txt'
            file_train.save(filepath_result)
            time.sleep(20)
            if os.path.exists('model1.pkl') and os.path.exists('model2.pkl'):
                print('exist')
                return('updatefinish')
        elif request.form['type'] == 'downloadFile':
            filename = request.form['filename']
            if os.path.exists(filename):
                print('exist')
                return send_file(filename)
        return 'none'


def NewModel():
    New = train_model.train()
    var3 = New.train_model(3)
    var1=New.train_model(1)
    var2=New.train_model(2)
    return var1,var2,var3

def Predict():
    pre = train_model.train()
    pre.predict()

def ApparentModel():
    Update = UseUpdate.try_update()
    Update.Try(1)
    Update.Try(2)

# def getFun(NonLinearRegressiondata,LinearRegressiondata):
#     str = ""
#     if LinearRegressiondata != "none":
#         for i in range(len(LinearRegressiondata)):
#             if LinearRegressiondata[i].Ltextit !="none":
#                 if LinearRegressiondata[i].Lvariable =="无":
#                     str = str + LinearRegressiondata[i].Ltextit
#                 elif LinearRegressiondata[i].Lvariable !="无":
#                     str = str + LinearRegressiondata[i].Ltextit+"*"+LinearRegressiondata[i].Lvariable
#     return str

#


#
#
if __name__ == '__main__':
    app.run()
