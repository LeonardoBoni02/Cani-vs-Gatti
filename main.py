import dataloader
import model
import train
import inference

train_data, test_data = dataloader.get_dataset(debug_samples=10,batch_size=2)

model_cnn = model.build_cnn_model()
model.show_model_summary(model_cnn)

train.train_model(train_data, test_data)

# Example inference
image_path = "C:/Users/tondi/Pictures/Screenshots/abdel.png"  # Replace with your image path

predicted_class = inference.infer_image(image_path)
print(f'Predicted class for the input image: {predicted_class}')