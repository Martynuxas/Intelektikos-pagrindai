import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow import keras

R = []
years = []
sunspots = []

def read():

    f = open('sunspot.txt', 'r')

    lines = f.readlines()
    for line in lines:
        temp=line.split('\t')
        years.append(int(temp[0]))
        sunspots.append(int(temp[1]))
    f.close()

    R.append(years)
    R.append(sunspots)
    return R
data = read()
print(data[1])

plt.plot(data[0], data[1], marker='o', markersize=4)
plt.title('Saulės dėmių grafikas 1700-2014m.')
plt.xlabel('Metai')
plt.ylabel('Saulės dėmių skaičius')
plt.show()
# --------------------------------------------------------------------------------------
n = 10
p = []
t = []

def split(data, n):
    sunspots = data[1]

    for i in range(len(sunspots) - n):
        temporary = []
        for j in range(i, i + n):
            temporary.append(sunspots[j])
        else:
            t.append((sunspots[j + 1]))
        p.append(temporary)
    R = list()
    R.append(p)
    R.append(t)
    return R

matrix = split(data, n)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = [i[0] for i in matrix[0]]
Y = [i[1] for i in matrix[0]]
print(matrix[0])
ax.scatter(X, Y, matrix[1])
ax.set_xlabel('Saulės dėmių skaičius(x-2) metais')
ax.set_ylabel('Saulės dėmių skaičius(x-1) metais')
ax.set_zlabel('Saulės dėmių skaičius x metais')

plt.show()

# TRAINING
L = 200
X = data[1]

model = LinearRegression().fit(matrix[0][:L-n], matrix[1][:L-n])
coef = model.coef_
w1 = coef[0]
w2 = coef[1]
b = model.intercept_
print('W1 {0} W2 {1} b {2}'.format(w1, w2, b))
predicted = model.predict(matrix[0][:L-n])

print('Real values: \n {0}'.format(data[1]))
print('Predicted: \n {0}'.format(predicted))
plt.title('Saulės dėmių grafikas {0}-{1}m'.format(1700 + n, 1900))
plt.xlabel('Metai')
plt.ylabel('Saulės dėmių skaičius')
plot1, = plt.plot(data[0][n:L], data[1][n:L], marker='o', markersize=4,
color='blue')
plot2, = plt.plot(data[0][n:L], predicted, marker='o', markersize=4,
color='red')
plt.legend([plot1,plot2],["Tikrosios reikšmės", "Prognozuotos reikšmės"])
plt.show()

# TESTING
length = len(data[0])
predicted = model.predict(matrix[0][:length-n])
plt.title('Saulės dėmių grafikas {0}-{1}m'.format(1700 + n, 2014))
plt.xlabel('Metai')
plt.ylabel('Saulės dėmių skaičius')
plot1, = plt.plot(data[0][n:length], data[1][n:length], marker='o',
markersize=4, color='blue')
plot2, = plt.plot(data[0][n:length], predicted, marker='o', markersize=4,
color='red')
plt.legend([plot1,plot2],["Tikrosios reikšmės", "Prognozuotos reikšmės"])
plt.show()

def errorVector(RR, PR, years):
    error = PR - RR

    print("Mediana su testiniais duomenim {0} {1}".format(years, PR.flatten()))

    plt.title('Testavimo ir prognozavimo klaidų dydžiai')
    plt.plot(years, error, marker='o', markersize=4, label='Testavimo ir prognozavimo klaidų dydžiai')

    plt.xlabel('Metai')
    plt.ylabel('Saulės dėmių skaičius')
    plt.show()

    return error

e = errorVector(data[1][n:length], predicted, data[0][n:length])

plt.hist(e)
plt.title('Prognozavimo klaidų histograma')
plt.xlabel('Klaidos reikšmė (skirtumas nuo tikrų duomenų)')

plt.ylabel('Dažnumas')
plt.show()

def MSE(count, error):
    mse_Sum = 0
    for i in error:
        mse_Sum += i * i

    value = 1 / count * mse_Sum

    return value

mse = MSE(length - n, e)
print('MSE {0}'.format(mse))

def MAD(error):
    mediana = np.median(np.absolute(error))
    return mediana

mad = MAD(e)
print('MAD {0}'.format(mad))
a = [[2,5], [3, 6]]
ats = np.dot(a, [5, 4])
print(ats)

#ITERACION METHOD
#TRAINING
X = matrix[0][:L-n]
Y = matrix[1][:L-n]
model = tf.keras.models.Sequential() # sukuria layer
model.add(tf.keras.layers.Dense(1, input_dim = n)) # kiek ivesciu
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer = opt, loss = 'mean_squared_error', metrics =
['mse'])
before = model.get_weights()
print("Svoriai prieš apmokymą: {0}".format(before))
history = model.fit(X, Y, epochs=200, batch_size=10, verbose=1)
mse = history.history['mse']
plt.plot(mse)
plt.xlabel("Iteracijos")
plt.ylabel("MSE")
plt.show()
after = model.get_weights()
print("Svoriai po apmokymo: {0}".format(after))
predictions = model.predict(X)
print("Spėjimai {0}".format(predictions))
plt.title('Saulės dėmių grafikas {0}-{1}m'.format(1700 + n, 1900))
plt.xlabel('Metai')
plt.ylabel('Saulės dėmių skaičius')

plot1, = plt.plot(data[0][n:L], data[1][n:L], marker='o', markersize=4,
color='blue')
plot2, = plt.plot(data[0][n:L], predictions, marker='o', markersize=4,
color='red')
plt.legend([plot1,plot2],["Tikrosios reikšmės", "Prognozuotos reikšmės"])
plt.show()
e = predictions - data[1][n:L]
mad = np.median(np.absolute(e))

#ITERATION METHOD
#TESTING
predictions = model.predict(matrix[0][:length-n])
plt.title('Saulės dėmių grafikas {0}-{1}m'.format(1700 + n, 2014))
plt.xlabel('Metai')
plt.ylabel('Saulės dėmių skaičius')
plot1, = plt.plot(data[0][n:length], data[1][n:length], marker='o',
markersize=4, color='blue')
plot2, = plt.plot(data[0][n:length], predictions, marker='o', markersize=4,
color='red')
plt.legend([plot1,plot2],["Tikrosios reikšmės", "Prognozuotos reikšmės"])
plt.show()
e = errorVector(data[1][n:length], predictions.flatten(),
data[0][n:length])
mse = MSE(length - n, e)
print('MSE {0}'.format(mse))
mad = MAD(e)
print('MAD {0}'.format(mad))