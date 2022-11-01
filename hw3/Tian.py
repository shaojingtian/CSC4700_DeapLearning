#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pylab', 'inline')


# In[2]:


import tensorflow as tf


# In[3]:


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)


# In[4]:


# -*- coding: utf-8 -*-
import numpy as np

def check_winner(L):
    N = len(L)
    if N < 5:
        return False
    else:
        s = np.sum(L[:5])
        if s == 5:
            return True
        if N > 5:
            for i in range(N-5):
                s = s - L[i] + L[i+5]
                if s == 5:
                    return True
        return False

class Board:
    def __init__(self, sz):
        self.sz = sz
        self.pbs = np.zeros((2, sz, sz), dtype=np.int)

    def add_move(self, p, x, y):
        self.pbs[p, x, y] = 1

        xd, xu = min(x, 4), min(self.sz-1-x, 4)
        yl, yr = min(y, 4), min(self.sz-1-y, 4)
        fs0, fs1 = min(xd, yl), min(xu, yr)
        bs0, bs1 = min(xu, yl), min(xd, yr)

        if check_winner(self.pbs[p, (x-xd):(x+xu+1), y]) or check_winner(self.pbs[p, x, (y-yl):(y+yr+1)]):
            return True
        elif check_winner(self.pbs[p, np.arange((x-fs0), (x+fs1+1)), np.arange((y-fs0), (y+fs1+1))]):
            return True
        elif check_winner(self.pbs[p, np.arange((x+bs0), (x-bs1-1), -1), np.arange((y-bs0), (y+bs1+1))]):
            return True
        else:
            return False


class Gomoku:
    def __init__(self, board_sz=11, gui=False):
        self.board_sz = board_sz
        self.board = Board(board_sz)
        self.number = np.zeros((board_sz, board_sz), dtype=int)
        self.k = 1  # step number
        self.result = 0
        if gui:
            self.gui = GameGUI(board_sz)
        else:
            self.gui = None


    def reset(self):
        self.board.pbs.fill(0)
        self.number.fill(0)
        self.k = 1
        self.result = 0

    def copy(self):  # copy the game, not the UI
        g = Gomoku(self.board_sz)
        g.board.pbs = np.copy(self.board.pbs)
        g.number = np.copy(self.number)
        g.k = self.k
        g.result = self.result
        return g

    def draw(self):
        print(self.board.pbs)
        if self.gui:
            self.gui._draw_background()
            self.gui._draw_chessman(self.board.pbs[0, :, :]-self.board.pbs[1, :, :], self.number)

    # execute a move
    def execute_move(self, p, x, y):
        assert np.sum(self.board.pbs[:, x, y]) == 0

        win = self.board.add_move(p, x, y)
        self.number[x][y] = self.k
        self.k += 1
        return win

    # main loop
    def play(self, p1, p2):
        players = {0:p1, 1:p2}
        pi = 0
        self.draw()
        while True:
            x, y = players[pi].get_move(self.board.pbs)
            if x < 0:
                break  
            win = self.execute_move(pi, x, y)
            self.draw()
            
            if win:
                self.result = 1-2*pi
                break
            if np.sum(self.board.pbs) == self.board_sz*self.board_sz:
                break
            
            pi = (pi+1) % 2



class RandomPlayer:
    def __init__(self, id):
        self.id = id

    def get_move(self, board):
        b = (board[0, :, :] + board[1, :, :]) - 1
        ix, jx = np.nonzero(b)
        idx = [i for i in zip(ix, jx)]
        return idx[np.random.choice(len(idx))]





    


# In[5]:


if __name__ == "__main__":
    
    # Two random player play 100 rounds of non-GUI game
        g = Gomoku()
        p1 = RandomPlayer(0)
        p2 = RandomPlayer(1)
        for i in range(100):
            g.play(p1, p2)
            print(i, g.result)
            g.reset()


# In[6]:


class MyPlayer:
    def __init__(self, id):
        self.id = id
        
    def get_move(self, board):
        board_sz=11
        self.board = Board(board_sz)
        board_config=np.moveaxis(self.board.pbs, 0, -1)
        input_shape = board_config.shape
        
        #ResNet Structure
        board_input = tf.keras.layers.Input(shape=input_shape)
        conv_input = tf.keras.layers.Conv2D(32,(5,5),padding='same')(board_input)
        batchnorm_input = tf.keras.layers.BatchNormalization()(conv_input)
        relu_input = tf.keras.layers.ReLU()(batchnorm_input)
        
        ConvA1_1 = tf.keras.layers.Conv2D(32,(5,5),padding='same')(relu_input)
        batchnormA1_1 = tf.keras.layers.BatchNormalization()(ConvA1_1)
        reluA1_1 = tf.keras.layers.ReLU()(batchnormA1_1)
        ConvA1_2 = tf.keras.layers.Conv2D(32,(5,5),padding='same')(reluA1_1)
        batchnormA1_2 = tf.keras.layers.BatchNormalization()(ConvA1_2)
        addA1 = tf.keras.layers.Add()([batchnormA1_2,relu_input])
        reluA1_2 = tf.keras.layers.ReLU()(addA1)
    
        ConvA2_1 = tf.keras.layers.Conv2D(32,(5,5),padding='same')(reluA1_2)
        batchnormA2_1 = tf.keras.layers.BatchNormalization()(ConvA2_1)
        reluA2_1 = tf.keras.layers.ReLU()(batchnormA2_1)
        ConvA2_2 = tf.keras.layers.Conv2D(32,(5,5),padding='same')(reluA2_1)
        batchnormA2_2 = tf.keras.layers.BatchNormalization()(ConvA2_2)
        addA2 = tf.keras.layers.Add()([batchnormA2_2,reluA1_2])
        reluA2_2 = tf.keras.layers.ReLU()(addA2)
    
        ConvA3_1 = tf.keras.layers.Conv2D(32,(5,5),padding='same')(reluA2_2)
        batchnormA3_1 = tf.keras.layers.BatchNormalization()(ConvA3_1)
        reluA3_1 = tf.keras.layers.ReLU()(batchnormA3_1)
        ConvA3_2 = tf.keras.layers.Conv2D(32,(5,5),padding='same')(reluA3_1)
        batchnormA3_2 = tf.keras.layers.BatchNormalization()(ConvA3_2)
        addA3 = tf.keras.layers.Add()([batchnormA3_2,reluA2_2])
        reluA3_2 = tf.keras.layers.ReLU()(addA3)
    
        ConvB1_1 = tf.keras.layers.Conv2D(64,(5,5),strides=(2,2),padding='same')(reluA3_2)
        batchnormB1_1 = tf.keras.layers.BatchNormalization()(ConvB1_1)
        reluB1_1 = tf.keras.layers.ReLU()(batchnormB1_1)
        ConvB1_2 = tf.keras.layers.Conv2D(64,(5,5),padding='same')(reluB1_1)
        batchnormB1_2 = tf.keras.layers.BatchNormalization()(ConvB1_2)
        skiptensorB1 = tf.keras.layers.Conv2D(64,(1,1),strides=(2,2),padding='same')(reluA3_2)
        addB1 = tf.keras.layers.Add()([batchnormB1_2,skiptensorB1])
        reluB1_2 = tf.keras.layers.ReLU()(addB1)
    
        ConvB2_1 = tf.keras.layers.Conv2D(64,(5,5),padding='same')(reluB1_2)
        batchnormB2_1 = tf.keras.layers.BatchNormalization()(ConvB2_1)
        reluB2_1 = tf.keras.layers.ReLU()(batchnormB2_1)
        ConvB2_2 = tf.keras.layers.Conv2D(64,(5,5),padding='same')(reluB2_1)
        batchnormB2_2 = tf.keras.layers.BatchNormalization()(ConvB2_2)
        addB2 = tf.keras.layers.Add()([batchnormB2_2,reluB1_2])
        reluB2_2 = tf.keras.layers.ReLU()(addB2)
    
        ConvB3_1 = tf.keras.layers.Conv2D(64,(5,5),padding='same')(reluB2_2)
        batchnormB3_1 = tf.keras.layers.BatchNormalization()(ConvB3_1)
        reluB3_1 = tf.keras.layers.ReLU()(batchnormB3_1)
        ConvB3_2 = tf.keras.layers.Conv2D(64,(5,5),padding='same')(reluB3_1)
        batchnormB3_2 = tf.keras.layers.BatchNormalization()(ConvB3_2)
        addB3 = tf.keras.layers.Add()([batchnormB3_2,reluB2_2])
        reluB3_2 = tf.keras.layers.ReLU()(addB3)
    
        ConvC1_1 = tf.keras.layers.Conv2D(128,(5,5),strides=(2,2),padding='same')(reluB3_2)
        batchnormC1_1 = tf.keras.layers.BatchNormalization()(ConvC1_1)
        reluC1_1 = tf.keras.layers.ReLU()(batchnormC1_1)
        ConvC1_2 = tf.keras.layers.Conv2D(128,(5,5),padding='same')(reluC1_1)
        batchnormC1_2 = tf.keras.layers.BatchNormalization()(ConvC1_2)
        skiptensorC1 = tf.keras.layers.Conv2D(128,(1,1),strides=(2,2),padding='same')(reluB3_2)
        addC1 = tf.keras.layers.Add()([batchnormC1_2,skiptensorC1])
        reluC1_2 = tf.keras.layers.ReLU()(addC1)
    
        ConvC2_1 = tf.keras.layers.Conv2D(128,(5,5),padding='same')(reluC1_2)
        batchnormC2_1 = tf.keras.layers.BatchNormalization()(ConvC2_1)
        reluC2_1 = tf.keras.layers.ReLU()(batchnormC2_1)
        ConvC2_2 = tf.keras.layers.Conv2D(128,(5,5),padding='same')(reluC2_1)
        batchnormC2_2 = tf.keras.layers.BatchNormalization()(ConvC2_2)
        addC2 = tf.keras.layers.Add()([batchnormC2_2,reluC1_2])
        reluC2_2 = tf.keras.layers.ReLU()(addC2)
    
        ConvC3_1 = tf.keras.layers.Conv2D(128,(5,5),padding='same')(reluC2_2)
        batchnormC3_1 = tf.keras.layers.BatchNormalization()(ConvC3_1)
        reluC3_1 = tf.keras.layers.ReLU()(batchnormC3_1)
        ConvC3_2 = tf.keras.layers.Conv2D(128,(5,5),padding='same')(reluC3_1)
        batchnormC3_2 = tf.keras.layers.BatchNormalization()(ConvC3_2)
        addC3 = tf.keras.layers.Add()([batchnormC3_2,reluC2_2])
        reluC3_2 = tf.keras.layers.ReLU()(addC3)
    
        #global_average_pooling
        pooling = tf.keras.layers.GlobalAveragePooling2D()(reluC3_2)
    
        #flatten
        flat = tf.keras.layers.Flatten()(pooling)
    
        #dense
        output = tf.keras.layers.Dense(121,activation='softmax')(flat)
    
        model = tf.keras.models.Model(inputs=board_input, outputs=output)
        
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        
        model.fit(board_config, board_config, verbose=1)
        
        model.save('Tian.h5')

        position = model.predict(board_config) 
        x = int(np.floor(position/11))
        y = position%11-1
        
        return x, y
    
    def get_model(self):
        m = tf.keras.models.load_model('Tian.h5')
        return m
    


# In[7]:


if __name__ == "__main__":
    
    # MyPlayer plays with random player 
        g = Gomoku()
        p1 = RandomPlayer(0)
        p2 = MyPlayer(1)
        for i in range(10):
            g.play(p1, p2)
            print(i, g.result)
            g.reset()


# In[ ]:




