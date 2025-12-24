import numpy as np
import matplotlib.pyplot as plt

Adam_test_loss = 'data/example_2_1/exact_test_loss_Adam.npy'
supervised_test_loss = 'data/section_2_3_2/supervised_test_loss_Adam.npy'


Adam_test_loss = np.load(Adam_test_loss)
supervised_test_loss = np.load(supervised_test_loss)


x= np.linspace(1,len(Adam_test_loss),len(Adam_test_loss))

plt.plot(x, Adam_test_loss, label= 'self supervised test loss', linestyle='-',)
plt.plot(x, supervised_test_loss,label='supervised test loss', linestyle='-')

plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.grid(True)
plt.savefig('section_2_3_2_loss.png')
plt.show()