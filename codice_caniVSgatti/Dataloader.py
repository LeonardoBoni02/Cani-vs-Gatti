from datasets import load_dataset
import numpy as np
import keras
import matplotlib.pyplot as plt 
#dataset cani vs gatti  
#class Dataloader:
class Dataloader:
    def __init__(self):
        self.dataset = load_dataset("microsoft/cats_vs_dogs")
        self.train_data = self.dataset['train']
        self.test_data = self.dataset['test']
    def get_data(self):
        return self.train_data, self.test_data
    def preprocess_data(self, data):
        images = []
        labels = []
        for item in data:
            image = np.array(item['image'].resize((128, 128))) / 255.0
            label = item['label']
            images.append(image)
            labels.append(label)
        return np.array(images), np.array(labels)
    def visualize_samples(self, data, num_samples=5):
        plt.figure(figsize=(15, 5))
        for i in range(num_samples):
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(data[i]['image'])
            plt.title('Dog' if data[i]['label'] == 1 else 'Cat')
            plt.axis('off')
        plt.show()
# Fine Dataloader.py


# Esempio di utilizzo
dataloader = Dataloader()
train_data, test_data = dataloader.get_data()
dataloader.visualize_samples(train_data)
X_train, y_train = dataloader.preprocess_data(train_data)
X_test, y_test = dataloader.preprocess_data(test_data)
print("Training data shape:", X_train.shape, y_train.shape)
print("Testing data shape:", X_test.shape, y_test.shape)

   