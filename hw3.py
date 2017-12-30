import hwlib as tr
import numpy as np
import tensorflow as tf


#===================================
#                arch helpers
#===================================

# добавляет сверточный слой с заданной формой        
def add_conv_layer(lr_num,x,K,N):
    form = K + [N]
    print('build lr:%i,K: %a' %(lr_num,form))
    initer = tf.initializers.variance_scaling
    with tf.variable_scope("conv_%i" %(lr_num)):
        W = tf.get_variable("W", shape=form)
        b = tf.get_variable("b", shape=[N])        
        a = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME') + b
        
        y = tf.nn.relu(a)
    return y

# добавляет pool/2 слой        
def add_pool_layer(lr_num,x):  
    print('build lr:%i' %(lr_num))
    with tf.variable_scope("pool_%i" %(lr_num)):
        y = tf.nn.max_pool(x,[1,2,2,1],strides=[1,2,2,1], padding='VALID')        
    return y

# добавляет полносвязный слой        
def add_dense_layer(lr_num,x, N = 512, add_relu = True) :
    sz = np.prod(x.shape[1:])
    print('build dense_lr:%i, flat_size=%a' %(lr_num,sz))
    initer = tf.initializers.variance_scaling
    with tf.variable_scope("dense_%i" %(lr_num)):
        W = tf.get_variable("W", shape=[sz, N])
        b = tf.get_variable("b", shape=[N])          
        #print('%i: %a' %(lr_num,x.shape))
        x_flat = tf.reshape(x,[-1,sz])
        a = tf.matmul(x_flat,W) + b        
        y = tf.nn.relu(a) if add_relu else a
            
    return y

#===================================
#                VGG like model
#===================================

def vgg_model():
    N = 10

    # placeholder'ы это точкb входа, можно восприпимать их, как аргументы функции, описываемой графом
    with tf.name_scope('inputs') :
        x = tf.placeholder(tf.float32, [None, 32, 32, 3],'x')
        y = tf.placeholder(tf.int64, [None],'y')
    train_mode = tf.placeholder(tf.bool,None,'mode')

    h = x    
    
    #========================== block #0
    ln = 1
    h = add_conv_layer(ln,h,[3, 3, 3], 64)
    ln +=1
    h = add_conv_layer(ln,h,[3, 3, 64], 64)
    h = add_pool_layer(1,h)
    print(h.shape)
    
    #========================== block #1
    ln +=1
    h = add_conv_layer(ln,h,[3, 3, 64], 128)
    ln +=1
    h = add_conv_layer(ln,h,[3, 3, 128], 128)
    h = add_pool_layer(2,h)
    print(h.shape)
    
    #========================== block #5
    ln +=1
    h = add_dense_layer(ln,h,256)
    print(h.shape)
    ln +=1
    h = add_dense_layer(ln,h,256)
    print(h.shape)
    ln +=1
    h = add_dense_layer(ln,h,10, False)
    print(h.shape)

    y_out = h    
    # y_out -- это вектор оценок, которые генерирует модель. Теперь определим функцию потерь
    total_loss = tf.losses.hinge_loss(tf.one_hot(y,N),logits=y_out)
    mean_loss = tf.reduce_mean(total_loss)
    
    correct_prediction = tf.equal(tf.argmax(y_out,1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    tf.summary.scalar('cost', total_loss)
    tf.summary.scalar('acc', accuracy)
    
    #Возвращаем те узлы графа, которые нам понадобятся в дальнейшем.
    #(x,y) это входы графа, а (y_out, mean_loss) выходы, которые представляют для нас интерес
    return (x,y), (y_out, mean_loss, accuracy)


import os
logs_path = os.getenv('RUN_PATH')
tr.run(vgg_model, logs_path)

#run(build_simple_model)

# vim:tw=78:enc=utf8
