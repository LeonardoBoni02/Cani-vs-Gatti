import keras
import model
import dataloader


def train_model(train_generator, val_generator):
    

    # Build the model
    cnn_model = model.build_cnn_model()

    # Train the model
    cnn_model.fit(train_generator,
                  validation_data=val_generator,
                  epochs=10)
    # Save the trained model in .keras
    cnn_model.save('app/cnn_cats_vs_dogs.keras')
    print("Model trained and saved as 'cnn_cats_vs_dogs.keras'")
    