# MNIST Digit Classification with Convolutional Neural Network

This project focuses on the task of digit classification using the MNIST dataset. The MNIST dataset is a popular benchmark dataset in the field of computer vision, consisting of a large collection of handwritten digit images.

In this project, we employ a Convolutional Neural Network (CNN) architecture to effectively classify the digits. CNNs have proven to be highly successful in image recognition tasks due to their ability to capture spatial dependencies through the use of convolutional layers.

## Dataset

The MNIST dataset contains 60,000 training images and 10,000 testing images. Each image is a grayscale 28x28 pixel image of a handwritten digit ranging from 0 to 9. The goal is to develop a model that can accurately classify these images into their corresponding digit labels.

## Model Architecture

The CNN model architecture used in this project comprises multiple layers to extract relevant features from the input images. The architecture consists of:

- Convolutional layers: These layers employ convolutional filters to detect important patterns and features in the images.
- Max pooling layers: These layers downsample the feature maps to retain the most significant information while reducing spatial dimensions.
- Fully connected layers: These layers connect all the neurons from the previous layers to the output layer, enabling the model to make predictions.

The specific architecture used is as follows:

1. Convolutional layer: 32 filters with a kernel size of 3x3.
2. Convolutional layer: 64 filters with a kernel size of 3x3.
3. Convolutional layer: 128 filters with a kernel size of 3x3.
4. Convolutional layer: 256 filters with a kernel size of 3x3.
5. Fully connected layer: 4096 neurons.
6. Fully connected layer: 50 neurons.
7. Output layer: 10 neurons (corresponding to the 10 possible digit classes).

## Training and Evaluation

The training process involves feeding the training images through the CNN model and optimizing the model's parameters using the stochastic gradient descent (SGD) algorithm. The model's performance is evaluated using the test set, and metrics such as accuracy and loss are calculated.

During training, the model's progress is tracked using the training and test losses, as well as the training and test accuracies. These metrics help monitor the model's performance and identify any potential issues such as overfitting or underfitting.

## Results

The trained model's performance is evaluated on the test set. The accuracy achieved on the test set indicates the model's ability to correctly classify the unseen digit images. Additionally, the average test loss provides insights into the model's overall predictive capability.

## Table of Contents

- [Installation](https://chat.openai.com/#installation)
- [Usage](https://chat.openai.com/#usage)
- [Contributing](https://chat.openai.com/#contributing)
- [License](https://chat.openai.com/#license)

## Installation

1. **Clone the repository:**

   ```
   https://github.com/prajnaraipn7/ERA_V1.git
   ```

2. **Change to the project directory:**

   ```
   cd repository-name
   ```

3. **Install the dependencies:**

   ```
   pip install torch torchvision tqdm matplotlib
   ```

## Usage

1. **Import the required modules in your Python script:**

   ```
   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   from torchvision import datasets, transforms
   from tqdm import tqdm
   from utils import *
   from model import Net
   ```

2. **Set up the device:**

   ```
   device = use_device()
   ```

3. **Define the transformations for training and testing data and make the data:**

   ```
   train_data,test_data = utils.make_data()
   ```

4. **Create the train and test data loaders:**

   ```
   kwargs = {'batch_size': 64, 'shuffle': True}
   train_loader, test_loader = data_loader(**kwargs)
   ```

5. **Create an instance of the model:**

   ```
   model = Net().to(device)
   ```

6. **Define the optimizer and the learning rate:**

   ```
   optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
   ```

7. **Train the model:**

   ```
   train_acc, train_losses = train(model, device, train_loader, optimizer)
   ```

8. **Test the model:**

   ```
   test_acc, test_losses = test(model, device, test_loader)
   ```

9. **Visualize the data:**

   ```
   batch_data, batch_label = next(iter(train_loader))
   plot_data(batch_data,batch_label,n_images)
   ```

10. **Set up the optimizer and scheduler:**

    ```
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)
    ```

11. **Train and test the model for multiple epochs:**

    ```
    num_epochs = 20
    
    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}')
        train_acc, train_losses = train(model, device, train_loader, optimizer)
        test_acc, test_losses = test(model, device, test_loader)
        scheduler.step()
    ```



## Conclusion

The MNIST Digit Classification project showcases the implementation of a Convolutional Neural Network for accurate digit classification. By leveraging the power of CNNs and the MNIST dataset, the model can learn and distinguish between different handwritten digits with high accuracy. 

## Contributing

Contributions are welcome! Here are some ways you can contribute:

- Fork the repository and make your changes.
- Open an issue to discuss potential changes or report a bug.
- Submit a pull request with your improvements.

## License

This project is licensed under the [MIT License](https://chat.openai.com/LICENSE). Feel free to use and modify the code according to your needs.

