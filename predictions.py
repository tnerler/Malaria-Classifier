from keras.models import load_model
from get_dataset import give_me_dataset
import matplotlib.pyplot as plt
import numpy as np

path = "C:/Users/tuana/Datasets/malaria/cell_images"

_, _, test_ds = give_me_dataset(path)

model = load_model("myModel.h5")
y_preds = model.predict(test_ds)


def plot_predictions(test_ds, y_preds, num_images=5):

    images, labels = next(iter(test_ds))  
    def get_str_labels(input): 
        return "Parasitized" if input == 0 else "Uninfected"

    
    def parasite_or_not(x) : 
        if x < 0.5 : 
            return "P"
        else :
            return 'U'

    
    if np.sqrt(num_images).is_integer(): 
        rows = int(np.sqrt(num_images))
        cols = rows
    else:
        print("The number of images must be a perfect square number.")
        return

    plt.figure(figsize=(10, 10))  
    
    for i in range(num_images):
        plt.subplot(rows, cols, i+1)  
        
        
        plt.imshow(images[i].numpy())  
        
        true_label = get_str_labels(labels.numpy()[i])
        predicted_label = parasite_or_not(model.predict(images)[i][0])
        
        plt.title(f"True: {true_label}\nPred: {predicted_label}")
        plt.axis('off')  

    plt.tight_layout() 
    plt.show()

plot_predictions(test_ds, y_preds, num_images=9)
