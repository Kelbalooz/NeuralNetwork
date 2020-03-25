import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt


EPOCHS = 1
BATCH_SIZE_TRAIN = 10
BATCH_SIZE_TEST = 1 #for this code, always stay 1
LEARNING_RATE = 0.00001
TRAIN_DATA_PATH = "./images/train/"
TEST_DATA_PATH = "./images/test/"

TRANSFORM_IMG = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True,  num_workers=4)
test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
test_data_loader  = data.DataLoader(test_data, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=4) 



class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(100352, 512)
        self.fc2 = nn.Linear(512, 1)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


TRAINING = True #put False for no training (duh)

if TRAINING:  
    if __name__ == '__main__':
    
        print("Number of train samples: ", len(train_data))
        print("Number of test samples: ", len(test_data))
        print("Detected Training classes are: ", train_data.class_to_idx) # classes are detected by folder structure
        print("Detected Testing classes are: ", test_data.class_to_idx) 
        
        model = CNN()    
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_func = nn.BCEWithLogitsLoss() 
        LISTLOSS=[]
        ACCURACY = []
    
        # Training and Testing
        for epoch in range(EPOCHS):
            for step, (x, y) in enumerate(train_data_loader):
                length=len(train_data_loader)
                progress=step/length *100
                b_x = Variable(x)   # batch x (image)
                b_y = Variable(y)   # batch y (target)
                output = model(b_x)
                b_y = b_y.type_as(output)
                loss = loss_func(output.squeeze(), b_y) 
                print("Progress: ", progress, "%:")
                print("Loss: ", loss.item())
                print("")
                LISTLOSS.append(loss)
                optimizer.zero_grad()           
                loss.backward()                 
                optimizer.step()
                if step % 500 == 0:
                    zero=0
                    print("Testing...")
                    good=0
                    total=0
                    for stepp, (a,b) in enumerate(test_data_loader):
                        print(stepp/len(test_data_loader))
                        print(good)
                        t_a = Variable(a)
                        t_b = Variable(b)
                        nrnianswer = model(t_a)
                        nranswer = Variable(model(t_a)).data.tolist()[0][0]
                        if nranswer<0.5:
                            answer=0
                        else:
                            answer=1
                        t_b = int(t_b)
                        '''
                        print("Answer not round not int: " , nrnianswer)
                        print("Answer not round:" , nranswer)
                        print("Answer: " , answer)
                        print("Actual Answer: " , t_b)
                        '''
                        if answer == t_b:
                            '''
                            print("Good")
                            '''
                            print("")
                            good+=1
                            total+=1
                        else:
                            total+=1
                            print("")
                    acc=good/total
                    ACCURACY.append(acc)
                    print("Accuracy: ", acc)
        print("Loss graph")                
        plt.plot(LISTLOSS)
        plt.show()
        print("Accuracy graph")
        plt.plot(ACCURACY)
        plt.show()
        torch.save(model, "WORLD_PROTOCOL2")
        print("Model saved")
                
    '''
                if step % 50 == 0:
                    test_x = Variable(test_data_loader)
                    test_output, last_layer = model(test_x)
                    pred_y = torch.max(test_output, 1)[1].data.squeeze()
                    accuracy = sum(pred_y == test_y) / float(test_y.size(0))
                    print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)
    '''

