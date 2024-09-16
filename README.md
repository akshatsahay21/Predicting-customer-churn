# Predicting-customer-churn
# Predicting Customer Churn For A Subscription Service
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample data [Age,Income,Purchased[1,0]]
data = np.array([
    [22,30,0],
    [25,40,0],
    [47,50,1],
    [52,80,1],
    [46,70,1],
    [23,20,0],
    [33,90,1],
    [35,100,1]
    ])

x = data[:,:2]
y = data[:,2]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)

model = LogisticRegression()
model.fit(x_train,y_train)
prediction = model.predict(x_test)
accuracy = accuracy_score(y_test,prediction)
print(f"accuracy_: {accuracy*100:.2f}%")
