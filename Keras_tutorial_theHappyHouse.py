# -*-coding:utf-8 -*-
"""
@project:untitled3
@author:JACK
@file:.py
@ide:untitled3
@time:2019-01-16 13:17:43
@month:一月
任务描述：下一次放假的时候，你决定和你的五个朋友一起度过一个星期。这里有一个非常好的房子任何想进入房子的人都必须证明他们目前的幸福状态。
  作为一个深度学习的专家，为了确保“快乐才开门”规则得到严格的应用，你将建立一个算法，它使用来自前门摄像头的图片来检查这个人是否快乐，
    只有在人高兴的时候，门才会打开。
"""
import numpy as np
from keras import layers
from keras.layers import Input,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D
from keras.layers import AveragePooling2D,MaxPooling2D,Dropout,GlobalMaxPooling2D,GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydoc
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *
import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

#######Normalize image vectors
X_train = X_train_orig / 255
X_test = X_test_orig / 255

Y_train = Y_train_orig.T
Y_test = Y_test_orig.T
print(X_train.shape[1:])
print("number of training example = " + str(X_train.shape[0]))
print("number of test example = " + str(X_test.shape[0]))
print("X_trian shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))
"""
images are of shape(64,64,3)
Training:600 pictures
Test:150 pictures
"""
####Building a model in Keras
"""Keras非常适合快速制作模型，它可以在很短的时间内建立一个很优秀的模型，举个例子:

def model(input_shape):
    X_input = Input(input_shape)##定义一个tensor的placeholder，维度为input_shape
    X = ZeroPadding2D((3,3))(X_input)###X_input周围填充0
    #####使用CONV--BN--RELU
    X = Conv2D(32,(7,7),strides=(1,1),name = 'conv0')(X)
    X = BatchNormalization(axis=3,name='bn0')(X)
    X = Activation('relu')(X)
    ###MAXPOOL
    X = MaxPooling2D((2,2),name='max_pool')(X)

    X = Flatten()(X)
    X = Dense(1,activation='sigmoid',name='fc')(X)
    model = Model(inputs = X_input,outputs = X,name = 'HappyModel')
    return model
"""
def HappyModel(input_shape):
    """
    Implemention of the HappyModel
    :param input_shape: shape of the image of the dataset
    :return: a Model() instance in Keras
    """
    #######start code here#####
    X_input = Input(input_shape)
    X = ZeroPadding2D((3,3))(X_input)

    #CONV--BN--RELU block applied to X
    X = Conv2D(32,(7,7),strides=(1,1),name = 'conv0')(X)
    X = BatchNormalization(axis=3,name = 'bn0')(X)
    X = Activation('relu')(X)

    # Maxpool
    X = MaxPooling2D((2,2),name = 'max_pool')(X)
    X = Flatten()(X)
    X = Dense(1,activation='sigmoid',name = 'fc')(X)
    model = Model(inputs = X_input, outputs = X, name = 'HappyModel')
    #########end code here#######
    return model

"""four step:Create -> Compile -> Fit(Train) -> Evaluate(Test)"""
####创建一个模型的实体
happy_model = HappyModel(X_train.shape[1:])#(64,64,3)
####编译模型
happy_model.compile('adam','binary_crossentropy',metrics=['accuracy'])
####训练模型
happy_model.fit(X_train,Y_train,epochs=1,batch_size=50)
"""Note：运行完40个epoch，继续使用fit的话，会继续使用已经训练好的参数，而不是重新开始训练"""
####评估模型
preds = happy_model.evaluate(X_test,Y_test,batch_size=32,verbose=1,sample_weight=None)

print("误差值 = " + str(preds[0]))
print("准确度 = " + str(preds[1]))
"""
误差值 = 0.10091055214405059
准确度 = 0.9666666642824808
"""
####真实的数据测试happy test
image_path = 'C:\\Users\\korey\\Desktop\\KunJ_happy.jpg'
img = image.load_img(image_path,target_size=(64,64))
imshow(img)
plt.show()

x = image.img_to_array(img)
x = np.expand_dims(x,axis = 0)
x = preprocess_input(x)
print(happy_model.predict(x))


































