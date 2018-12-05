#coding: utf-8
import matplotlib.pyplot as plt
import pandas as pd

train = pd.read_csv('train.txt',sep=',')
val = pd.read_csv('test.txt',sep=',')

plt.figure()
plt.plot(train['NumIters'], train['softmax_loss'], label = 'train_loss')
plt.xlabel('Iterations')
plt.ylabel('loss')
plt.title('train loss vs Iters')
plt.savefig('train_loss.jpg')

plt.figure()
plt.plot(val['NumIters'], val['softmax_loss'], label = 'val_loss')
plt.xlabel('Iterations')
plt.ylabel('loss')
plt.title('val loss vs Iters')
plt.savefig('val_loss.jpg')