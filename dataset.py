import pandas as pd 
from torch.utils.data import Dataset
import os 
import imageio as io
from sklearn.model_selection import train_test_split
import numpy as np 

class LeafDataset(Dataset): 
    INDEX_TO_CLASS = {
        "0": "Cassava Bacterial Blight (CBB)", 
        "1": "Cassava Brown Streak Disease (CBSD)", 
        "2": "Cassava Green Mottle (CGM)", 
        "3": "Cassava Mosaic Disease (CMD)", 
        "4": "Healthy"
        }

    CLASS_TO_INDEX = {item:key for key, item in INDEX_TO_CLASS.items()}

    def __init__(self, csv_file, root_dir, mode='train' ,transform = None, random_state = 42, num_classes = 5) -> None:
        super(LeafDataset).__init__()
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir    
        self.transform = transform
        self.train_df , self.test_df = train_test_split(self.data, test_size=0.1, stratify=self.data.label, random_state=random_state)
        self.train_df.reset_index(inplace = True, drop = True)
        self.test_df.reset_index(inplace = True, drop = True)
        self.mode = mode
        self.num_classes = num_classes 
        if mode == 'train':
            self.class_weights_tensor = self.class_weights(self.train_df)
        
    def __len__(self): 

        if self.mode == 'train':
            return len(self.train_df)
        elif self.mode == 'test': 
            return len(self.test_df)
        else: 
            raise Exception("mode should be either train or test")

    def __getitem__(self, index): 
        
        if self.mode == 'train': 
            image_name = self.train_df.iloc[index,0]
            label = self.train_df.iloc[index,1]
        else: 
            image_name = self.test_df.iloc[index,0]
            label = self.test_df.iloc[index,1]

        image_path = os.path.join(self.root_dir, image_name)
        sample = io.imread(image_path)

        if self.transform: 
            sample = self.transform(sample)

        return (sample, label)

    def class_weights (self, data_fram):
        # Get the number of images in each class in the training set. 
        classes_counts_dict= pd.value_counts(self.train_df.label).to_dict()
        classes_counts_sorted = {i : classes_counts_dict[i] for i in range(self.num_classes)}
        classes_array = 1./(np.array(list(classes_counts_sorted.values()))/len(self.train_df))
        normalized_classes_array = classes_array / classes_array.sum()
        return normalized_classes_array