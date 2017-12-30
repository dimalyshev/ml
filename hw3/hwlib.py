
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split

from random import randrange

def init_train() :
    #4
    plt.style.use('ggplot')
    #%matplotlib inline
    plt.rcParams['figure.figsize'] = (15, 12) # set default size of plots

    #7
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    #8
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    #10
    #Preprocessing: вычитаем среднее
    # 1: Находим среднее изображение
    mean_image = np.mean(x_train, axis=0)
    plt.figure(figsize=(4,4))
    plt.imshow(mean_image.reshape((32,32,3)).astype('uint8')) # визуализируем полученное среднее
    #visualize mean image
    #plt.show()

    #11
    # 2: вычитаем среднее из изображений обучающей и тестовых выборок
    x_train = x_train - mean_image
    x_test = x_test - mean_image

    #12
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=1000)

    return (x_train, x_val, y_train, y_val, x_test, y_test)

#9
def plot_classes() :
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 10
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(x_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    #visualize classes
    plt.show()

#plot_classes()

#28
#===================================
#                simple model
#===================================

def build_simple_model():
    # placeholder'ы это точкb входа, можно восприпимать их, как аргументы функции, описываемой графом
    x = tf.placeholder(tf.float32, [None, 32, 32, 3],'in')
    y = tf.placeholder(tf.int64, [None],'out')
    
    #variable scope задаёт префикс для всех элементов внутри него
    #Это позволяет огранизовавывать структуру графа и вашего кода
    with tf.variable_scope("convolution_layer_1"):
        #создаём веса (W -- ядра свёрток, b -- сдвиг)
        Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
        bconv1 = tf.get_variable("bconv1", shape=[32])
        
        a1 = tf.nn.conv2d(x, Wconv1, strides=[1,2,2,1], padding='VALID') + bconv1
        h1 = tf.nn.relu(a1)
    
    print(h1.shape)
    #добавляем полносвязный слой
    with tf.variable_scope("dense_layer_1"):
        W1 = tf.get_variable("W1", shape=[5408, 10])
        b1 = tf.get_variable("b1", shape=[10])
        
        h1_flat = tf.reshape(h1,[-1,5408])
        y_out = tf.matmul(h1_flat,W1) + b1

        
    # y_out -- это вектор оценок, которые генерирует модель. Теперь определим функцию потерь
    total_loss = tf.losses.hinge_loss(tf.one_hot(y,10),logits=y_out)
    mean_loss = tf.reduce_mean(total_loss)
    
    correct_prediction = tf.equal(tf.argmax(y_out,1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    #Возвращаем те узлы графа, которые нам понадобятся в дальнейшем.
    #(x,y) это входы графа, а (y_out, mean_loss) выходы, которые представляют для нас интерес
    return (x,y), (y_out, mean_loss, accuracy)


#===================================
#                training_loop
#===================================
                              
def training_loop(merged, logs_path, session, model_inputs, model_outputs, train_step, epochs=10, batch_size=64):

    (x_train, x_val, y_train, y_val, x_test, y_test) = init_train()
    
    #создаём индекс по всем объектам
    index = np.arange(len(x_train))
    
    #перемешиваем его
    np.random.shuffle(index)
    
    #разбиваем на батчи
    num_batches = int(len(index) / batch_size)
    batch_indexes = np.array_split(index, num_batches)
    
    #аналогично для теста
    index_test = np.arange(len(x_test))
    np.random.shuffle(index_test)
    num_batches_test = int(len(index_test) / batch_size)
    batch_indexes_test = np.array_split(index_test, num_batches_test)
    
    #аналогично для validation
    index_val = np.arange(len(x_val))
    np.random.shuffle(index_val)
    num_batches_val = int(len(index_val) / batch_size)
    batch_indexes_val = np.array_split(index_val, num_batches_val)
    
    
    x,y = model_inputs
    y_out, mean_loss, accuracy = model_outputs
    
    def train(log_path, epoch, x_values, y_values, batch_indexes):
        train_loses = []

        sz = epoch * len(batch_indexes)
        wr = tf.summary.FileWriter(log_path, session.graph)

        for i, batch_index in enumerate(batch_indexes): 
            #Создаём словарь, осуществляющий сопоставление входов графа (plaseholders) и значений 
            feed_dict = {x: x_values[batch_index],
                         y: y_values[batch_index]}

            #Здесь происходит непоследственный вызов модели
            #Обратите внимание, что мы передаём train_step
            scores, loss, acc, _, summ = session.run([y_out, mean_loss, accuracy, train_step, merged],feed_dict=feed_dict)

            if i%20 == 0 :
                wr.add_summary(summ,i+sz)

            train_loses.append(loss)
            print(f'iteration {i}, train loss: {loss:.3}, accuracy: {acc:.3}', end='\r')
        return train_loses
        
    def evaluate(log_path, epoch, x_values, y_values, batch_indexes):
        test_loses = []
        test_accuracy = []

        wr = tf.summary.FileWriter(log_path, session.graph)
        sz = epoch * len(batch_indexes)

        for i, batch_index in enumerate(batch_indexes) :

            #Создаём словарь, осуществляющий сопоставление входов графа (plaseholders) и значений
            feed_dict = {x: x_values[batch_index],
                     y: y_values[batch_index]}

            #Здесь происходит непоследственный вызов модели
            loss, acc, summ = session.run([mean_loss, accuracy, merged],feed_dict=feed_dict)

            test_loses.append(loss)
            test_accuracy.append(acc)

            if i%10 == 0 :
                wr.add_summary(summ,i+sz)

        return test_loses, test_accuracy
       
    # цикл по эпохам
    for e in range(epochs):
       print(f'Epoch {e}:')
       train_loses = train(logs_path+'\\train', e, x_train, y_train, batch_indexes)
       val_loses, val_accuracy = evaluate(logs_path+'\\val', e, x_val, y_val, batch_indexes_val)
       print(f'train loss: {np.mean(train_loses):.3}, val loss: {np.mean(val_loses):.3}, accuracy: {np.mean(val_accuracy):.3}')
     
    print('================================================')
    print('Test set results:')
    test_loses, test_accuracy = evaluate(logs_path+'\\test', 0, x_test, y_test, batch_indexes_test)
    print(f'test loss: {np.mean(test_loses):.3}, accuracy: {np.mean(test_accuracy):.3}')

#===================================
#                launcher
#===================================

def run(model, logs_path) :
    #Перед вызовом функции очистим память от графов других моделей (актуально если вы вызываете эту ячейку повторно)
    tf.reset_default_graph()
    (x,y), (y_out, mean_loss, accuracy) = model()

    #Теперь зададим алгоритм оптимизации
    optimizer = tf.train.AdamOptimizer(5e-5, name = 'adam') 
    #train_step -- специальный служебный узел в графе, отвечающий за обратный проход
    train_step = optimizer.minimize(mean_loss) 

    merged = tf.summary.merge_all()
    # создаём сессию. Сессия -- это среда, в которой выполняются вычисления
    with tf.Session() as sess:
        #мы можем явно указать устройство
        with tf.device("/cpu:0"): #"/cpu:0" or "/gpu:0" 

            #инициализируем веса, в этот момент происходит выделение памяти
            sess.run(tf.global_variables_initializer())

            #запускаем тренировку
            training_loop(merged, logs_path, sess, model_inputs=(x,y), 
                          model_outputs=(y_out, mean_loss, accuracy), 
                          train_step=train_step, epochs=20)


