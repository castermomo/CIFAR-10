#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import cifar10
import numpy as np
from tensorflow.keras.utils import to_categorical
np.random.seed(10)


# In[2]:


(x_img_train, y_label_train), (x_img_test, y_label_test) = cifar10.load_data()


# In[3]:


x_img_train_normalize=x_img_train.astype('float32')/255.0
x_img_test_normalize=x_img_test.astype('float32')/255.0


# In[4]:


y_label_train_onehot =tf.keras.utils. to_categorical(y_label_train)
y_label_test_onehot = tf.keras.utils.to_categorical(y_label_test)


# In[5]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Dropout,Flatten,MaxPooling2D,Activation,ZeroPadding2D


# In[6]:


model = Sequential()
model.add(Conv2D(filters=32,
               kernel_size=(3,3),
                 padding='same',
                 input_shape=(32,32,3),
               activation='relu'))
model.add(Dropout(rate=0.25))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=64,
               kernel_size=(3,3),
                 padding='same',
               activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=128,
               kernel_size=(4,4),
                 padding='same',
               activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(rate=0.25))
model.add(Dense(1024,activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(10,
               activation='softmax'))
print(model.summary())


# In[7]:


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[8]:


train_history=model.fit(x_img_train_normalize,
                        y_label_train_onehot,
                       validation_split=0.2,
                       epochs=10,batch_size=128,verbose=1)


# In[9]:


import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()


# In[10]:


show_train_history(train_history,'accuracy','val_accuracy')


# In[11]:


show_train_history(train_history,'loss','val_loss')


# In[12]:


scores = model.evaluate(x_img_test_normalize,y_label_test_onehot,verbose=0)
scores[1]


# In[13]:


prediction=model.predict_classes(x_img_test_normalize)


# In[14]:


prediction[:10]


# In[15]:


label_dict={0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}


# In[16]:


import matplotlib.pyplot as plt
def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num>25: num=25
    for i in range(0,num):
        ax=plt.subplot(5,5,1+i)
        ax.imshow(images[idx],cmap='binary')
        title=str(i)+','+label_dict[labels[i][0]]
        if len(prediction)>0:
            title+="=>"+label_dict[prediction[i]]
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        idx+=1
    plt.show()


# In[17]:


plot_images_labels_prediction(x_img_test,y_label_test,prediction,0,10)


# In[18]:


Predicted_Probability=model.predict(x_img_test_normalize)
def show_Predicted_Probability(y,prediction,x_img,Predicted_Probability,i):
    print('label:',label_dict[y[i][0]],
         'predict',label_dict[prediction[i]])
    plt.figure(figsize=(2,2))
    plt.imshow(np.reshape(x_img_test[i],(32,32,3)))
    plt.show
    for j in range(10):
        print(label_dict[j]+'Probability:%1.9f'%(Predicted_Probability[i][j]))


# In[19]:


show_Predicted_Probability(y_label_test,prediction,x_img_test,Predicted_Probability,0)


# In[20]:


show_Predicted_Probability(y_label_test,prediction,x_img_test,Predicted_Probability,3)


# In[21]:


prediction.shape


# In[22]:


y_label_test.shape


# In[23]:


y_label_test.reshape(-1)


# In[24]:


import pandas as pd
print(label_dict)


# In[25]:


pd.crosstab(y_label_test.reshape(-1),prediction,rownames=['label'],colnames=['predict'])

