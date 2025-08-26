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
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)  
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


```
```python

Parthiban = PeopleClassifier(input_size=x_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(Parthiban.parameters(),lr=0.01)
```
```python

def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
      model.train()
      for X_batch,y_batch in train_loader:
        optimizer.zero_grad()
        outputs=model(X_batch)
        loss=criterion(outputs,y_batch)
        loss.backward()
        optimizer.step()


    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    #Include your code here
```



## Dataset Information

<img width="1286" height="245" alt="image" src="https://github.com/user-attachments/assets/5a609845-a649-4689-894d-34555153f7dd" />


## OUTPUT



### Confusion Matrix

<img width="732" height="575" alt="image" src="https://github.com/user-attachments/assets/49aa0780-6fc2-4670-9427-74d0753e88c4" />



### Classification Report

<img width="595" height="354" alt="image" src="https://github.com/user-attachments/assets/60825f3a-09eb-40fa-926d-f51c52ebe31b" />




### New Sample Data Prediction

<img width="565" height="94" alt="image" src="https://github.com/user-attachments/assets/485781e3-7951-4f16-95ba-1ac040948927" />


## RESULT
Thus, a neural network classification model for the given dataset as been created successfully.
