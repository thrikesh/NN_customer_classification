# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="1004" height="842" alt="image" src="https://github.com/user-attachments/assets/63c35434-2547-4bfc-8fd6-eb5046aedc30" />


## DESIGN STEPS

## STEP 1:
Import necessary libraries and load the dataset.

## STEP 2:
Encode categorical variables and normalize numerical features.

## STEP 3:
Split the dataset into training and testing subsets.

## STEP 4:
Design a multi-layer neural network with appropriate activation functions.

## STEP 5:
Train the model using an optimizer and loss function.

## STEP 6:
Evaluate the model and generate a confusion matrix.

## STEP 7:
Use the trained model to classify new data samples.

## STEP 8:
Display the confusion matrix, classification report, and predictions.

## PROGRAM

### Name: THRIKESWAR
### Register Number:212222230162

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)
    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return x


```
```python

model =PeopleClassifier(input_size=X_train.shape[1])
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)
```
```python

def train_model(model,train_loader,criterion,optimizer,epochs):
  for epoch in range(epochs):
    model.train()
    for X_batch,y_batch in train_loader:
      optimizer.zero_grad()
      outputs=model(X_batch)
      loss=criterion(outputs,y_batch)
      loss.backward()
      optimizer.step()

  if(epoch+1)%10==0:
    print(f'Epoch [{epoch+1}/{epochs}],Loss:{loss.item():.4f}')
```



## Dataset Information
<img width="1237" height="237" alt="image" src="https://github.com/user-attachments/assets/59d6ff97-59c3-4b86-8139-61f836bf3e08" />



## OUTPUT



### Confusion Matrix
<img width="677" height="568" alt="image" src="https://github.com/user-attachments/assets/95a577c0-c405-468b-a117-a93562fb5976" />




### Classification Report

<img width="145" height="44" alt="image" src="https://github.com/user-attachments/assets/f599dabd-4f77-49b5-8773-a18916e6fe5e" />





### New Sample Data Prediction

<img width="348" height="95" alt="image" src="https://github.com/user-attachments/assets/09f40df6-b90a-4940-a883-819aa5a28565" />



## RESULT
Thus, a neural network classification model for the given dataset as been created successfully.
