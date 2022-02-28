import matplotlib.pyplot as plt
import pickle

with open("raw/app1_training_history", "rb") as f:
    history = pickle.load(f)

plt.figure()
[plt.plot(history[k]['val_loss']) for k in range(5)]
plt.legend([f'fold {k}' for k in range(5)], loc='upper right')
plt.title('Loss of Neural Network on Validation Set')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.savefig('results/1_valid_loss.eps')
