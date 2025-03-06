import matplotlib.pyplot as plt
import pandas as pd

history = pd.read_csv("C:/Users/tuana/My Projects/Malaria Detection/training_log.csv")

train_loss = history['loss']
train_acc = history['accuracy']
val_loss = history['val_loss']
val_acc = history['val_accuracy']


plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_loss) + 1), train_loss, label='Train Loss')
plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_acc) + 1), train_acc, label='Train Accuracy')
plt.plot(range(1, len(val_acc) + 1), val_acc, label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
