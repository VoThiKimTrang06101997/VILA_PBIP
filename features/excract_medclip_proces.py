import os
import torch
from PIL import Image
import pickle as pkl
from torchvision import transforms
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor
from tqdm import tqdm
import cv2 as cv
from utils.pyutils import set_seed
from albumentations.pytorch import ToTensorV2
import albumentations as A

# Defines image augmentations and normalization. These are standard practices to make the model robust
def get_transform():
    MEAN = [0.66791496, 0.47791372, 0.70623304] # Mean values for normalization, specific to the dataset
    STD = [0.1736589,  0.22564577, 0.19820057] # Standard deviation values.
    
    transform = A.Compose([
        A.Normalize(MEAN, STD), # Normalizes the image using the specified mean and std
        A.HorizontalFlip(p=0.5), # Randomly flips the image horizontally
        A.VerticalFlip(p=0.5), # Randomly flips the image vertically
        A.RandomRotate90(), # Randomly rotates the image by 90 degrees
        ToTensorV2(transpose_mask=True), # Converts the image to a PyTorch tensor
    ])
    return transform

def extract_features(image_dir):
    set_seed(42) 

    # Loads the MedCLIP model with a Vision Transformer (ViT) backbone
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = model.to(device) 
    model.eval()
    
    features_dict = {} # Initializes a dictionary to store features for each class
    # Iterates over the four tissue classes. The paper uses BCSS dataset which has these classes
    for class_name in ['NEC', 'LYM', 'STR', 'TUM']:
        class_path = os.path.join(image_dir, class_name) # Path to the images for the current class
        features_dict[class_name] = [] # Initializes a list for the current class's features
        
        # Gets a list of all image files in the class directory
        image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        print(f"\nProcessing {class_name} images...")
        # Iterates through each image file with a progress bar
        for img_name in tqdm(image_files, desc=f"{class_name}", ncols=100):
            img_path = os.path.join(class_path, img_name)
            img = cv.imread(img_path, cv.IMREAD_UNCHANGED)

            transform = get_transform() # Gets the augmentation pipeline
            img = transform(image=img)["image"] # Applies the transformations
            # Normalizes the tensor to a [0, 1] range for stability
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            with torch.no_grad(): # Disables gradient calculation for inference
                # Feeds the image to the MedCLIP vision model and gets the output features
                outputs = model.vision_model(img.unsqueeze(0).to(device))
                features = outputs.cpu().detach().numpy() # Moves features to CPU, detaches from the graph, and converts to a NumPy array
            
            # Appends the image name and its extracted features to the dictionary
            features_dict[class_name].append({
                'name': img_name,
                'features': features
            })
    
    save_dir = "./features/image_features" # Directory to save the output file
    os.makedirs(save_dir, exist_ok=True) # Creates the directory if it doesn't exist
    save_path = os.path.join(save_dir, "bcss_features_pro.pkl") # Defines the output file path
    
    with open(save_path, 'wb') as f:
        pkl.dump(features_dict, f)
    
    print(f"\nFeatures saved to {save_path}")

if __name__ == "__main__":
    image_dir = "data/BCSS-WSSS/proto" # Path to the directory containing prototype images, sorted into class folders
    extract_features(image_dir)