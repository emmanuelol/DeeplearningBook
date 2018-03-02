import numpy as np
np.random.seed(3)

dWax = np.random.randn(5,3)*10
dWaa = np.random.randn(5,5)*10
dWya = np.random.randn(2,5)*10
db = np.random.randn(5,1)*10
dby = np.random.randn(2,1)*10

gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "db": db, "dby": dby}

dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']

for gradient in [dWax, dWaa, dWya, db, dby]:
    np.clip(gradient,-10,10,gradient)
    print(gradient)

print("Waa:",gradients['dWaa'])
