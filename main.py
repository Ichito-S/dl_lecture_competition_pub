import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm
from torchvision import transforms, models, datasets
from scipy.signal import resample, butter, filtfilt
from sklearn.model_selection import train_test_split
import shutil

# Utility functions
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def preprocess_signal(signal, resample_rate=250, lowcut=0.5, highcut=50.0, fs=1000, order=5):
    # Resampling
    num_samples = int(len(signal) * resample_rate / fs)
    signal = resample(signal, num_samples)
    
    # Butterworth bandpass filter
    nyquist = 0.5 * resample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    signal = filtfilt(b, a, signal)
    
    # Baseline correction (mean removal)
    signal -= np.mean(signal)
    
    # Scaling (standardization)
    signal /= np.std(signal)
    
    return signal

# Custom dataset class with signal preprocessing
class CustomMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split, data_dir):
        self.data_dir = os.path.join(data_dir, split)
        self.samples = os.listdir(self.data_dir)
        self.signal_transform = preprocess_signal  # Custom signal preprocessing

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_path = os.path.join(self.data_dir, self.samples[index])
        signal = np.load(sample_path)
        label = int(self.samples[index].split('_')[0])
        signal = self.signal_transform(signal)
        return signal, label, index

# Function to get image dataloader
def get_image_dataloader(image_dir, batch_size, num_workers):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(image_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names

# Function to train the model
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        print()

    return model

# Custom model using the fine-tuned ResNet for feature extraction
class ResNetEncoder(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super(ResNetEncoder, self).__init__()
        self.feature_extractor = pretrained_model
        self.feature_extractor.fc = nn.Identity()  # Remove the final classification layer
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
        output = self.fc(features)
        return output

# Main function
@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Ensure the correct directory structure for image data
    image_dir = "C:/dl_lecture_competition_pub/Images"
    classes = os.listdir(image_dir)
    train_dir = os.path.join(image_dir, 'train')
    val_dir = os.path.join(image_dir, 'val')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for class_name in classes:
        class_path = os.path.join(image_dir, class_name)
        if os.path.isdir(class_path) and class_name not in ['train', 'val']:
            images = os.listdir(class_path)

            if len(images) > 1:
                train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)
            else:
                train_images, val_images = images, []

            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

            for image in train_images:
                shutil.move(os.path.join(class_path, image), os.path.join(train_dir, class_name, image))

            for image in val_images:
                shutil.move(os.path.join(class_path, image), os.path.join(val_dir, class_name, image))

            if not os.listdir(class_path):  # Check if the directory is empty before removing
                shutil.rmtree(class_path)

    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader for images
    # ------------------
    dataloaders, dataset_sizes, class_names = get_image_dataloader(image_dir, batch_size=32, num_workers=4)

    # ------------------
    #       Model
    # ------------------
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, num_epochs=25)

    # ------------------
    #    Dataloader for MEG data
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    
    train_set = CustomMEGDataset("train", args.data_dir)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = CustomMEGDataset("val", args.data_dir)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = CustomMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Initialize the custom model for MEG data
    model = ResNetEncoder(model_ft, num_classes=train_set.num_classes).to(device)

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(device)
      
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        model.train()
        for X, signal, y, subject_idxs in tqdm(train_loader, desc="Train"):
            X, signal, y = X.to(device), signal.to(device), y.to(device)

            y_pred = model(X)  # Modified model forward pass
            
            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        model.eval()
        for X, signal, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, signal, y = X.to(device), signal.to(device), y.to(device)
            
            with torch.no_grad():
                y_pred = model(X)  # Modified model forward pass
            
            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        
        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)
            
    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=device))

    preds = [] 
    model.eval()
    for X, signal, subject_idxs in tqdm(test_loader, desc="Validation"):        
        preds.append(model(X.to(device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()
