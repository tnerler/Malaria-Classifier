from get_dataset import give_me_dataset
from model import cnn_model

path = "C:/Users/tuana/Datasets/malaria/cell_images"

train_ds, val_ds, test_ds = give_me_dataset(path)

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger

model = cnn_model()

callbacks = [
    # ModelCheckpoint: Save the best model
    ModelCheckpoint(
        "myModel.h5",               
        monitor='val_loss',            
        save_best_only=True,           
        mode='min',                    
        verbose=1
    ),
    
    # EarlyStopping: Stop training if validation loss does not improve
    EarlyStopping(
        monitor='val_loss',            
        patience=5,                    
        mode='min',                    
        verbose=1
    ),
    
    # ReduceLROnPlateau: Reduce learning rate if validation loss does not improve
    ReduceLROnPlateau(
        monitor='val_loss',           
        factor=0.2,                   
        patience=3,                    
        min_lr=1e-6,                   
        verbose=1
    ),
    
    # TensorBoard: Visualize the training process
    TensorBoard(
        log_dir='./logs',              
        histogram_freq=1,              
        write_graph=True,             
        write_images=True,             
        update_freq='epoch'   )  ,      
    
    # CSVLogger: Log training metrics into a CSV file
    CSVLogger(
        'training_log.csv',            
        separator=',',                 
        append=False )                  
    ]

history = model.fit(train_ds, batch_size=32, epochs=10, callbacks=callbacks, validation_data=val_ds)


loss, acc = model.evaluate(test_ds)
print('Model Test Loss : ', loss)
print("Model Test Acc : ", acc)
