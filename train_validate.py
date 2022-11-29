import torch 
from tqdm import tqdm 
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

class Train_Validate(): 
    
    def __init__(self, model, train_loader, test_loader, epochs, optimizer, criterion, device) -> None:

        self.train_loader = train_loader   
        self.test_loader = test_loader
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.model = model
        self.device = device

    def fit_model(self): 
        training_losses = []
        training_accs = [] 
        f1s = []
        self.model.train()
        self.model.to(self.device)
        for epoch in range(self.epochs):
            print(f'Epoch {epoch+1} / {self.epochs}')
            training_loss = 0.0
            training_acc = 0.0 
            f1_predictions = []
            f1_labels = []
            for images, labels in tqdm(self.train_loader):
                self.optimizer.zero_grad()
                images, labels = images.to(self.device), labels.to(self.device)
                probs = self.model(images)
                loss = self.criterion(probs, labels)
                predictions = torch.argmax(probs, dim=-1)
                acc = (predictions == labels).sum().item() / len(labels)
                f1_predictions.append(predictions.detach().cpu())
                f1_labels.append(labels.detach().cpu())
                training_acc += acc 
                training_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            training_losses.append(training_loss/len(self.train_loader))
            training_accs.append(training_acc/len(self.train_loader))
            f1 = f1_score(y_true = torch.flatten(torch.stack(f1_labels)), y_pred= torch.flatten(torch.stack(f1_predictions)), average = 'micro')
            f1s.append(f1)
            conf_matrix = confusion_matrix(
                y_true = torch.flatten(torch.stack(f1_labels)), 
                y_pred = torch.flatten(torch.stack(f1_predictions)), 
                normalize = 'true'
                )
            print(
                f'Training Loss = {(training_loss/len(self.train_loader)):.3f} \t Training acc = {(training_acc/len(self.train_loader))*100:.3f} \t F1_score = {f1:.3f}'
                )
            print(f'Confusion Matrix \n {conf_matrix}')
            print(' =========================================================== ')
        return training_accs, training_losses, f1s
    
    def evaluation(self): 
        self.model.eval()
        test_loss = 0.0 
        test_acc = 0.0 
        f1s_labels = []
        f1s_predictions = []
        with torch.no_grad(): 
            for images, labels in tqdm(self.test_loader): 
                images, labels = images.to(self.device), labels.to(self.device)
                probs = self.model(images)
                predictions = torch.argmax(probs, dim=-1)
                loss = self.criterion(probs, labels)
                test_loss += loss.item()
                acc = (predictions == labels).sum().item()/len(labels)
                f1s_predictions.append(predictions.detach().cpu())
                f1s_labels.append(labels.detach().cpu())
                test_acc += acc
            f1 = f1_score(
                y_true= torch.flatten(torch.stack(f1s_labels)),
                y_pred= torch.flatten(torch.stack(f1s_predictions)),
                average = 'micro'
                )
            conf_matrix = confusion_matrix(
                y_true = torch.flatten(torch.stack(f1s_labels)), 
                y_pred = torch.flatten(torch.stack(f1s_predictions)), 
                normalize = 'true'
                )
            print(f'Test Loss = {(test_loss/len(self.test_loader)):.3f} \t Test Acc = {(test_acc/len(self.test_loader)):.3f} \t Test F1_score = {f1:.3f}')
            print(f'Confusion Matrix \n {conf_matrix}')
        return test_loss, test_acc, f1