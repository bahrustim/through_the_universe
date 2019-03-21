import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers

#Импортируем данные из файлов в датафрэйм 
X_df = pd.read_csv("X_data.csv", sep=";")
X_df = X_df[5:]
Y_df = pd.read_csv("Y_train.csv", header=None, sep=";")
Y_submit_df = pd.read_csv("Y_submit.csv", header=None, sep=";")

#Переводим датафреймы в массивы Numpy.
#Обучающие+тестовые данные будут иметь размер (29184+5808,60,18). (пакеты, время, признаки)
X_train_size = X_df.shape[0]//60
X_train = X_df[:X_train_size*60].values.reshape(X_train_size, 60, 18)
X_train = X_train[71:]
Y_train = Y_df.values
Y_submit = Y_submit_df.values

#Отбросим колонку с отметкой времени. Теперь размер будет (29184+5808,60,17)
X_train = X_train[:,:,1:] 
Y_train = Y_train[:,1:]

#Заменим нулями данные не влияющие на результат. Например, в конце обжига образец
#находится в пятой камере, а данные о температуре первой камеры в этот момент вносят только шум в обучающие данные.
X_train[:,13:,:3] = 0
X_train[:,:13,3:6] = 0
X_train[:,25:,3:6] = 0
X_train[:,:25,6:9] = 0
X_train[:,37:,6:9] = 0
X_train[:,:37,9:12] = 0
X_train[:,49:,9:12] = 0
X_train[:,:49,12:15] = 0

#Отделим тестовые данные от обучающих
X_submit = X_train[X_train.shape[0]-Y_submit.shape[0]:]
X_train = X_train[:X_train.shape[0]-Y_submit.shape[0]]

#Проверяем размеры входных данных в нейронную сеть
X_train.shape
Y_train.shape
Y_submit.shape
X_submit.shape

#Создаем последовательную модель
model = models.Sequential()

#Добавляем уровни сети. Выбираем для обучения сверточную одномерную модель, так как
#она учитывает пордок в котором идут данные в каждом единичном обучающем  пакете, а это важно в нашей задаче
model.add(layers.Conv1D(16, 3, activation = 'relu', input_shape = (60, 17)))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(32, 3, activation = 'relu'))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(64, 3, activation = 'relu'))
model.add(layers.MaxPooling1D(2))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation = 'relu'))
model.add(layers.Dense(256, activation = 'relu'))
model.add(layers.Dense(1))
model.summary()

# Компилируем модель. Выбираем оптимизатор adam, потому что он хорошо подходит для задачи регрессии
model.compile(loss="mae", optimizer="adam")

# Обучаем сеть. Оптимальное количество эпох при котором не наступает переобучения - 7.
model.fit(X_train, Y_train, batch_size=64, epochs=7, shuffle = True, validation_split=0.1)

# Предсказываем результаты для тестовых данных. Записываем данные в файл
prediction = model.predict(X_submit)
prediction = np.round(prediction.flatten())
Y_submit_df.loc[:, 1] = (prediction.astype('int16'))
Y_submit_df.to_csv("new_submit.csv", index = False, header=False)

# Сравниваем полученный резльтат на тестовых данных с результатом на обучающих данных
plt.plot(Y_train[::5], label = 'Train data')
plt.plot(Y_submit_df.values[:,1], label = 'Test data')
plt.legend()
plt.show()

# Сравним входные данные для минимального результата
Y_train[np.argmin(Y_train)]
plt.plot(X_train[np.argmin(Y_train)])
plt.show()

Y_submit_df.values[np.argmin(Y_submit_df.values[:,1]),1]
plt.plot(X_submit[np.argmin(Y_submit_df.values[:,1])])
plt.show()



# Сравним входные данные для максимального результата
Y_train[np.argmax(Y_train)]
plt.plot(X_train[np.argmax(Y_train)])
plt.show()

Y_submit_df.values[np.argmax(Y_submit_df.values[:,1]),1]
plt.plot(X_submit[np.argmax(Y_submit_df.values[:,1])])
plt.show()

