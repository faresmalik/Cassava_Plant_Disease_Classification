import argparse
import torch
from dataset import LeafDataset
from train_validate import Train_Validate
from torchvision import transforms
from torch.utils.data import DataLoader
import timm 
import os

parser = argparse.ArgumentParser(description='Train ViT')

#needed for dataset/dataloader creation 
    ## Non-default values ##
parser.add_argument('--batch_size', type=int ,
                    help='Choose The Size of the Batch [16,32,64,128]')
parser.add_argument('--epochs', type=int ,
                    help='Number of Epochs')


    ## Default values ## 
parser.add_argument('--image_size', type=int , default = 224,
                    help='Image Size')
parser.add_argument('--csv_file_path', type=str , default='train.csv',
                    help='Pass CSV File Path. default: train.csv')
parser.add_argument('--train_images_path', type=str , default='train_images',
                    help='Train Images Directory Path')
parser.add_argument('--random_state', type=int , default = 42,
                    help='Random State for Splitting Data')
parser.add_argument('--shuffle_train', default = True,
                    help='Shuffle Training Data When Creating DataLoader')
parser.add_argument('--shuffle_test', default = False,
                    help='Shuffle Testing Data When Creating DataLoader')
parser.add_argument('--num_workers', type=int , default = 12,
                    help='Num Workers')
parser.add_argument('--transforms_prob', type=float , default = 0.5,
                    help='Transformations Probability')
parser.add_argument('--transform_degrees', type=int , default = 2,
                    help='Transformations Probability')
parser.add_argument('--test_size', type=float , default = 0.1,
                    help='Test Size Split')
parser.add_argument('--num_classes', type=int , default = 5,
                    help='Number of Classes')
parser.add_argument('--model_name', type=str , default= 'vit_base_patch16_224_in21k',
                    help='ViT model')

#needed for training the model 
parser.add_argument('--class_weights', default = True,
                    help='Include Class Weight in The Loss')

parser.add_argument('--lr', type=float , default= 1e-5,
                    help='Learning Rate')

#If you have multiple GPUs, specify the gpu number.
parser.add_argument('--gpu_number', type=int , default= 0,
                    help='Learning Rate')

args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']= str(args.gpu_number)

print('Creating Dataset and DataLoader')
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p = args.transforms_prob),
    transforms.RandomResizedCrop(size = (args.image_size, args.image_size), scale = (0.95,1.0)),
    transforms.RandomVerticalFlip(p= args.transforms_prob),
    transforms.RandomAffine(degrees= args.transform_degrees),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor()
])

train_dataset = LeafDataset(
    csv_file = args.csv_file_path , root_dir = args.train_images_path, 
    transform = train_transform, mode= 'train', random_state= args.random_state,
    num_classes= args.num_classes
    )

test_dataset = LeafDataset(
    csv_file = args.csv_file_path , root_dir = args.train_images_path, 
    transform = test_transform, mode= 'test', random_state= args.random_state,
    num_classes= args.num_classes
    )

train_dataloader = DataLoader(
    dataset= train_dataset,batch_size = args.batch_size,
    shuffle = args.shuffle_train,num_workers= args.num_workers,drop_last= True
)

test_dataloader = DataLoader(
    dataset= test_dataset,batch_size = args.batch_size,
    shuffle = args.shuffle_test,num_workers= args.num_workers,drop_last= True, 
)

print('Load Model for Training')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = timm.create_model(args.model_name, pretrained= True, num_classes = args.num_classes).to(device)

optimizer = torch.optim.Adam(params= model.parameters(), lr = args.lr)

if args.class_weights: 
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(train_dataset.class_weights_tensor, dtype = torch.float).to(device))
    print(f'Class weights are included in the loss function: \n {train_dataset.class_weights_tensor}')
else: 
    criterion = torch.nn.CrossEntropyLoss()
    print('Class weights are NOT included in the loss function')

leaves_classification_model = Train_Validate(
    model = model, train_loader = train_dataloader,
    test_loader= test_dataloader, epochs = args.epochs, optimizer = optimizer,
    criterion = criterion, device = device
)

print('================ Start Training ====================')
train_acc, train_loss, f1_train = leaves_classification_model.fit_model()
print('================ Training Finished! ====================')

print('================ Start Evaluation ====================')
leaves_classification_model.evaluation()
print('================ Evaluation Finished! ====================')
