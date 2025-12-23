import numpy as np
import matplotlib.pyplot as plt

SGD_train_loss = 'data/example_2_1/exact_train_loss_SGD.npy'
SGD_test_loss = 'data/example_2_1/exact_test_loss_SGD.npy'
Adam_train_loss = 'data/example_2_1/exact_train_loss_Adam.npy'
Adam_test_loss = 'data/example_2_1/exact_test_loss_Adam.npy'


SGD_train_loss = np.load(SGD_train_loss)
SGD_test_loss = np.load(SGD_test_loss)
Adam_train_loss = np.load(Adam_train_loss)
Adam_test_loss = np.load(Adam_test_loss)


x= np.linspace(1,len(SGD_train_loss),len(SGD_train_loss))

plt.plot(x, SGD_train_loss, label= 'SGD Training loss', linestyle='-',)
plt.plot(x, SGD_test_loss,label='SGD Test loss', linestyle='-')
plt.plot(x, Adam_train_loss, label= 'Adam Training loss', linestyle='-',)
plt.plot(x, Adam_test_loss,label='Adam Test loss', linestyle='-')

plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.grid(True)
plt.savefig('example_2_1_loss.png')
plt.show()