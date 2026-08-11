import os
import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from model import CNN

def train_model():
    print("Setting up training for Cats vs Dogs model...")
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Path to dataset (relative to parent directory)
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'PetImages'))
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"PetImages dataset folder not found at: {dataset_path}. Please make sure it is in the root project folder.")

    # Load dataset
    print(f"Loading images from {dataset_path}...")
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    print(f"Dataset loaded. Class to Index mapping: {dataset.class_to_idx}")

    # Splits (70% train, 15% val, 15% test)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    print(f"Total images: {total_size} | Train size: {train_size} | Val size: {val_size} | Test size: {test_size}")

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    # Data Loaders
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters())

    # Training loop
    epochs = 10
    print(f"Starting training on CPU for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        epoch_training_loss = 0.0

        for batch_idx, (images, labels) in enumerate(trainloader):
            optimiser.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimiser.step()

            epoch_training_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(trainloader):
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(trainloader)}], Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for images, labels in valloader:
                output = model(images)
                loss = criterion(output, labels)
                val_running_loss += loss.item()

        avg_train_loss = epoch_training_loss / len(trainloader)
        avg_val_loss = val_running_loss / len(valloader)
        print(f"--- Epoch {epoch+1} finished: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f} ---")

    # Evaluation on Test Dataset
    print("Evaluating model on test dataset...")
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted_val = torch.max(outputs, 1)
            correct += (predicted_val == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Evaluation complete. Total test samples: {total} | Correct: {correct} | Accuracy: {accuracy:.4f}")

    # Save weights
    save_path = os.path.join(os.path.dirname(__file__), 'cats_dogs_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Successfully saved model weights to: {save_path}")

if __name__ == '__main__':
    train_model()
