import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader # Import DataLoader


def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # Set shuffle based on training or test
    shuffle_data = False
    if training:
        shuffle_data = True

    # Load the dataset
    dataset = datasets.FashionMNIST(
        './data', 
        train=training, 
        download=True, 
        transform=transform
    )
    
    # Create the DataLoader
    loader = DataLoader(
        dataset, 
        batch_size=64, 
        shuffle=shuffle_data
    )
    
    return loader



def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),  # Input layer to hidden layer 1
        nn.ReLU(),
        nn.Linear(128, 64),       # Hidden layer 1 to hidden layer 2
        nn.ReLU(),
        nn.Linear(64, 10)         # Hidden layer 2 to output classes layer
    )
    
    return model



def build_deeper_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256), # Input layer to hidden layer 1 
        nn.ReLU(),
        nn.Linear(256, 128),     # Hidden layer 1 to hidden layer 2
        nn.ReLU(),
        nn.Linear(128, 64),      # Hidden layer 2 to hidden layer 3
        nn.ReLU(),
        nn.Linear(64, 32),       # Hidden layer 3 to hidden layer 4 
        nn.ReLU(),
        nn.Linear(32, 10)        # Hidden layer 4 to output layer
    )
    
    return model 



def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model.train()
    
    # Loop over the number of epochs
    for epoch in range(T):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Loop over the training data batches
        for images, labels in train_loader:
            # Zero the parameter gradients
            opt.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Optimize
            opt.step()
            
            # Update running loss
            running_loss += loss.item() * images.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        # Calculate average loss and accuracy for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total
        
        print(f"Train Epoch: {epoch} Accuracy: {correct}/{total}({epoch_acc:.2f}%) Loss: {epoch_loss:.3f}")
    

    


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()
    
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        # Loop over the test data batches
        for images, labels in test_loader:
            # Forward pass
            outputs = model(images)
            
            # Calculate the loss
            loss = criterion(outputs, labels)
            
            # Update total loss
            test_loss += loss.item() * images.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    # Calculate average loss and accuracy
    avg_loss = test_loss / len(test_loader.dataset)
    avg_acc = 100 * correct / total
    
    if show_loss:
        print(f"Average loss: {avg_loss:.4f}")
    print(f"Accuracy: {avg_acc:.2f}%")

    


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
    
    # Get the specific image
    image = test_images[index]
    
    # Add a batch dimension
    image = image.unsqueeze(0)
    
    # Set to evaluation mode
    model.eval()
    
    # Disable gradient calculations
    with torch.no_grad():
        # get model output and convert to probabilities
        logits = model(image)
        probabilities = F.softmax(logits, dim=1)
        
        # Get the top 3 probabilities
        top_probs, top_indices = torch.topk(probabilities, 3)
        
    # Squeeze tensors to remove the batch dimension for easier handling
    top_probs = top_probs.squeeze()
    top_indices = top_indices.squeeze()
    
    # Print the top 3 predictions
    for i in range(3):
        prob = top_probs[i].item() * 100
        # Explicitly cast to int to resolve linter warning
        label_index = int(top_indices[i].item()) 
        class_name = class_names[label_index]
        print(f"{class_name}: {prob:.2f}%")



if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()

    # Test get_data_loader
    print("Test get_data_loader...")
    train_loader = get_data_loader(training=True)
    test_loader = get_data_loader(training=False)
    print(f"Train loader type: {type(train_loader)}")
    print(f"Test loader type: {type(test_loader)}")
    assert hasattr(train_loader.dataset, '__len__')
    assert hasattr(test_loader.dataset, '__len__')

    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
    
    # Test build_model
    print("\nTest build_model")
    model = build_model()
    print(model)
    
    # Test build_deeper_model
    print("\nTest build_deeper_model")
    deeper_model = build_deeper_model()
    print(deeper_model)

    # Test train_model
    print("\nTest train_model")
    train_model(model, train_loader, criterion, T=5)

    # Test evaluate_model
    print("\nTest evaluate_model")
    evaluate_model(model, test_loader, criterion, show_loss=True)
    
    # Test predict_label
    print("\nTest predict_label")
    test_images, _ = next(iter(test_loader))
    predict_label(model, test_images, 0)