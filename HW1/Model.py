#!/usr/bin/env python
# coding: utf-8

# In[13]:


#https://drive.google.com/file/d/19vBnBnLGAAx9jMRdvJGg4c0exKROsbiB/view?usp=sharing


# In[14]:


get_ipython().run_line_magic('pylab', 'inline')


# In[15]:


import tensorflow as tf


# In[16]:


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)


# In[17]:


def create_model(x_train):
    # input
    input_shape = x_train.shape
    visible = tf.keras.layers.Input(shape=input_shape[1:])
    conv_input = tf.keras.layers.Conv2D(32,(3,3),padding='same')(visible)
    batchnorm_input = tf.keras.layers.BatchNormalization()(conv_input)
    relu_input = tf.keras.layers.ReLU()(batchnorm_input)
    
    #section A
    #A1
    ConvA1_1 = tf.keras.layers.Conv2D(32,(3,3),padding='same')(relu_input)
    batchnormA1_1 = tf.keras.layers.BatchNormalization()(ConvA1_1)
    reluA1_1 = tf.keras.layers.ReLU()(batchnormA1_1)
    ConvA1_2 = tf.keras.layers.Conv2D(32,(3,3),padding='same')(reluA1_1)
    batchnormA1_2 = tf.keras.layers.BatchNormalization()(ConvA1_2)
    addA1 = tf.keras.layers.Add()([batchnormA1_2,relu_input])
    reluA1_2 = tf.keras.layers.ReLU()(addA1)
    
    #A2
    ConvA2_1 = tf.keras.layers.Conv2D(32,(3,3),padding='same')(reluA1_2)
    batchnormA2_1 = tf.keras.layers.BatchNormalization()(ConvA2_1)
    reluA2_1 = tf.keras.layers.ReLU()(batchnormA2_1)
    ConvA2_2 = tf.keras.layers.Conv2D(32,(3,3),padding='same')(reluA2_1)
    batchnormA2_2 = tf.keras.layers.BatchNormalization()(ConvA2_2)
    addA2 = tf.keras.layers.Add()([batchnormA2_2,reluA1_2])
    reluA2_2 = tf.keras.layers.ReLU()(addA2)
    
    #A3
    ConvA3_1 = tf.keras.layers.Conv2D(32,(3,3),padding='same')(reluA2_2)
    batchnormA3_1 = tf.keras.layers.BatchNormalization()(ConvA3_1)
    reluA3_1 = tf.keras.layers.ReLU()(batchnormA3_1)
    ConvA3_2 = tf.keras.layers.Conv2D(32,(3,3),padding='same')(reluA3_1)
    batchnormA3_2 = tf.keras.layers.BatchNormalization()(ConvA3_2)
    addA3 = tf.keras.layers.Add()([batchnormA3_2,reluA2_2])
    reluA3_2 = tf.keras.layers.ReLU()(addA3)
    
    #section B
    #B1
    ConvB1_1 = tf.keras.layers.Conv2D(64,(3,3),strides=(2,2),padding='same')(reluA3_2)
    batchnormB1_1 = tf.keras.layers.BatchNormalization()(ConvB1_1)
    reluB1_1 = tf.keras.layers.ReLU()(batchnormB1_1)
    ConvB1_2 = tf.keras.layers.Conv2D(64,(3,3),padding='same')(reluB1_1)
    batchnormB1_2 = tf.keras.layers.BatchNormalization()(ConvB1_2)
    skiptensorB1 = tf.keras.layers.Conv2D(64,(1,1),strides=(2,2),padding='same')(reluA3_2)
    addB1 = tf.keras.layers.Add()([batchnormB1_2,skiptensorB1])
    reluB1_2 = tf.keras.layers.ReLU()(addB1)
    
    #B2
    ConvB2_1 = tf.keras.layers.Conv2D(64,(3,3),padding='same')(reluB1_2)
    batchnormB2_1 = tf.keras.layers.BatchNormalization()(ConvB2_1)
    reluB2_1 = tf.keras.layers.ReLU()(batchnormB2_1)
    ConvB2_2 = tf.keras.layers.Conv2D(64,(3,3),padding='same')(reluB2_1)
    batchnormB2_2 = tf.keras.layers.BatchNormalization()(ConvB2_2)
    addB2 = tf.keras.layers.Add()([batchnormB2_2,reluB1_2])
    reluB2_2 = tf.keras.layers.ReLU()(addB2)
    
    #B3
    ConvB3_1 = tf.keras.layers.Conv2D(64,(3,3),padding='same')(reluB2_2)
    batchnormB3_1 = tf.keras.layers.BatchNormalization()(ConvB3_1)
    reluB3_1 = tf.keras.layers.ReLU()(batchnormB3_1)
    ConvB3_2 = tf.keras.layers.Conv2D(64,(3,3),padding='same')(reluB3_1)
    batchnormB3_2 = tf.keras.layers.BatchNormalization()(ConvB3_2)
    addB3 = tf.keras.layers.Add()([batchnormB3_2,reluB2_2])
    reluB3_2 = tf.keras.layers.ReLU()(addB3)
    
    #section C
    #C1
    ConvC1_1 = tf.keras.layers.Conv2D(128,(3,3),strides=(2,2),padding='same')(reluB3_2)
    batchnormC1_1 = tf.keras.layers.BatchNormalization()(ConvC1_1)
    reluC1_1 = tf.keras.layers.ReLU()(batchnormC1_1)
    ConvC1_2 = tf.keras.layers.Conv2D(128,(3,3),padding='same')(reluC1_1)
    batchnormC1_2 = tf.keras.layers.BatchNormalization()(ConvC1_2)
    skiptensorC1 = tf.keras.layers.Conv2D(128,(1,1),strides=(2,2),padding='same')(reluB3_2)
    addC1 = tf.keras.layers.Add()([batchnormC1_2,skiptensorC1])
    reluC1_2 = tf.keras.layers.ReLU()(addC1)
    
    #C2
    ConvC2_1 = tf.keras.layers.Conv2D(128,(3,3),padding='same')(reluC1_2)
    batchnormC2_1 = tf.keras.layers.BatchNormalization()(ConvC2_1)
    reluC2_1 = tf.keras.layers.ReLU()(batchnormC2_1)
    ConvC2_2 = tf.keras.layers.Conv2D(128,(3,3),padding='same')(reluC2_1)
    batchnormC2_2 = tf.keras.layers.BatchNormalization()(ConvC2_2)
    addC2 = tf.keras.layers.Add()([batchnormC2_2,reluC1_2])
    reluC2_2 = tf.keras.layers.ReLU()(addC2)
    
    #C3
    ConvC3_1 = tf.keras.layers.Conv2D(128,(3,3),padding='same')(reluC2_2)
    batchnormC3_1 = tf.keras.layers.BatchNormalization()(ConvC3_1)
    reluC3_1 = tf.keras.layers.ReLU()(batchnormC3_1)
    ConvC3_2 = tf.keras.layers.Conv2D(128,(3,3),padding='same')(reluC3_1)
    batchnormC3_2 = tf.keras.layers.BatchNormalization()(ConvC3_2)
    addC3 = tf.keras.layers.Add()([batchnormC3_2,reluC2_2])
    reluC3_2 = tf.keras.layers.ReLU()(addC3)
    
    #global_average_pooling
    pooling = tf.keras.layers.GlobalAveragePooling2D()(reluC3_2)
    
    #flatten
    flat = tf.keras.layers.Flatten()(pooling)
    
    #dense
    output = tf.keras.layers.Dense(10,activation='softmax')(flat)
    
    model = tf.keras.models.Model(inputs=visible, outputs=output)
    
    return model
    
    


# In[18]:


cifar10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# In[19]:


model_1 = create_model()
model_1.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model_1.fit(x_train,y_train, epochs=50, batch_size=100)
val_loss, val_acc = model_1.evaluate(x_test, y_test,verbose=2)
print(val_loss,val_acc)


# In[20]:


model_1.save('my_model.h5')


# In[21]:


def get_trained_model():
    loaded_model = tf.keras.models.load_model('my_model.h5')
    return loaded_model


# In[22]:


model_2 = get_trained_model()
prediction = model_2.predict([x_test])
print(prediction)


# In[ ]:




