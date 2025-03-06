import tensorflow as tf
def give_me_dataset(path) : 

    dataset = tf.keras.preprocessing.image_dataset_from_directory(path, 
                                                                  label_mode="int", 
                                                                  batch_size=32, 
                                                                  image_size=(128, 128), 
                                                                  seed=42)
    def split(dataset, TRAIN_RATIO, VAL_RATIO, TEST_RATIO) : 

        dataset = dataset.shuffle(1000, seed=42)
        DATASET_SIZE = len(dataset)

        train_ds = dataset.take(int(TRAIN_RATIO * DATASET_SIZE))
        val_test_ds = dataset.skip(int(TRAIN_RATIO * DATASET_SIZE))

        val_ds = val_test_ds.take(int(VAL_RATIO * DATASET_SIZE))
        test_ds = val_test_ds.skip(int(VAL_RATIO * DATASET_SIZE))

        return train_ds, val_ds, test_ds
    
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1

    train_ds, val_ds, test_ds = split(dataset, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)


    normalization_layer = tf.keras.layers.Rescaling(1./255)
    
    def rescale(image, label) : 
        return normalization_layer(image), label
    
    train_dataset = train_ds.map(rescale)
    val_dataset = val_ds.map(rescale)
    test_dataset = test_ds.map(rescale)

    return train_dataset, val_dataset, test_dataset

    
