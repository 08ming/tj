import os
import random
import numpy as np
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.metrics import top_k_categorical_accuracy
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.utils import to_categorical

class_names_to_ids = {'cardboard': 0, 'glass': 1, 'metal': 2, 'paper':3, 'plastic':4, 'trash':5}

data_dir = './dataset/'
output_path = 'list.txt'
fd = open(output_path, 'w')
for class_name in class_names_to_ids.keys():
    images_list = os.listdir(data_dir + class_name)
    for image_name in images_list:
        fd.write('{}/{} {}\n'.format(class_name, image_name, class_names_to_ids[class_name]))
fd.close()

# 随机选取样本做训练集和测试集
_NUM_VALIDATION = 505
_RANDOM_SEED = 0
list_path = 'list.txt'
train_list_path = 'list_train.txt'
val_list_path = 'list_val.txt'
fd = open(list_path)
lines = fd.readlines()
fd.close()
random.seed(_RANDOM_SEED)
random.shuffle(lines)
fd = open(train_list_path, 'w')
for line in lines[_NUM_VALIDATION:]:
    fd.write(line)
fd.close()
fd = open(val_list_path, 'w')
for line in lines[:_NUM_VALIDATION]:
    fd.write(line)
fd.close()

def get_train_test_data(list_file):
    list_train = open(list_file)
    x_train = []
    y_train = []
    for line in list_train.readlines():
        x_train.append(line.strip()[:-2])
        y_train.append(int(line.strip()[-1]))
        #print(line.strip())
    return x_train, y_train
x_train, y_train = get_train_test_data('list_train.txt')
x_test, y_test = get_train_test_data('list_val.txt')

def process_train_test_data(x_path):
    images = []
    for image_path in x_path:
        img_load = load_img('./dataset/'+image_path)
        img = image.img_to_array(img_load)
        img = preprocess_input(img)
        images.append(img)
    return images
train_images = process_train_test_data(x_train)
test_images = process_train_test_data(x_test)

from keras.applications.inception_resnet_v2 import InceptionResNetV2
base_model = InceptionResNetV2(include_top=False, pooling='avg')
outputs = Dense(6, activation='softmax')(base_model.output)
model = Model(base_model.inputs, outputs)

# 设置ModelCheckpoint，按照验证集的准确率进行保存
save_dir = './train_model/'
filepath = "model_{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), monitor='val_acc', verbose=1,
                             save_best_only=True)
# 模型设置
def acc_top3(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def acc_top5(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', acc_top3, acc_top5])
 #模型训练
model.fit(np.array(train_images), to_categorical(y_train),
          batch_size=8,
          epochs=1,
          shuffle=True,
          validation_data=(np.array(test_images), to_categorical(y_test)),
          callbacks=[checkpoint])

