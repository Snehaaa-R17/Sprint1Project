#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Start the water quality analysis task by importing the necessary Python libraries and the dataset:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

data = pd.read_csv("water_potability.csv")
data.head(29000)


# In[2]:


data.describe()


# In[3]:


#There are null values in the first preview of this dataset itself, so before we go ahead, let’s remove all the rows that contain null values:

data = data.dropna()
data.isnull().sum()


# In[4]:


data.isnull().mean().plot.bar(figsize=(10,6)) 
plt.ylabel('Percentage of missing values') 
plt.xlabel('Features') 
plt.title('Missing Data in Percentages');


# In[5]:


data.info()


# In[6]:


data.shape # Check the shape of data


# In[7]:


data.dtypes # checking data types


# In[8]:


# Count Distribution of target variable
sns.countplot(x = data["Potability"])
plt.title("Distribution of Unsafe and Safe Water")

print(f"{data.Potability[data.Potability==1].count() / data.Potability.count()*100:.2f} % of samples are potable (1)")

#The Potability column of this dataset is the column we need to predict because it contains values 0 and 1 that indicate whether the water is potable (1) or unfit (0) for consumption.


# In[9]:


# Correlation matrix for dataset
Corrmat = data.corr()
plt.subplots(figsize=(7,7))
sns.heatmap(Corrmat, cmap="YlGnBu", square = True, annot=True, fmt='.2f')
plt.show()


# In[10]:


# Distribution of features
import plotly.express as px
data = data
figure = px.histogram(data, x = "ph", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: PH")
figure.show()

#The ph column represents the ph value of the water which is an important factor in evaluating the acid-base balance of the water. The pH value of drinking water should be between 6.5 and 8.5.


# In[11]:


figure = px.histogram(data, x = "Hardness", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Hardness")
figure.show()

#The figure above shows the distribution of water hardness in the dataset. The hardness of water usually depends on its source, but water with a hardness of 120-200 milligrams is drinkable.


# In[12]:


figure = px.histogram(data, x = "Solids", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Solids")
figure.show()

#The figure above represents the distribution of total dissolved solids in water in the dataset. All organic and inorganic minerals present in water are called dissolved solids. Water with a very high number of dissolved solids is highly mineralized. 


# In[13]:


figure = px.histogram(data, x = "Chloramines", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Chloramines")
figure.show()

#The figure above represents the distribution of chloramine in water in the dataset. Chloramine and chlorine are disinfectants used in public water systems. 


# In[14]:


figure = px.histogram(data, x = "Sulfate", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Sulfate")
figure.show()

#The figure above shows the distribution of sulfate in water in the dataset. They are substances naturally present in minerals, soil and rocks. Water containing less than 500 milligrams of sulfate is safe to drink.


# In[15]:


figure = px.histogram(data, x = "Conductivity", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Conductivity")
figure.show()

#The figure above represents the distribution of water conductivity in the dataset. Water is a good conductor of electricity, but the purest form of water is not a good conductor of electricity. Water with an electrical conductivity of less than 500 is drinkable. 


# In[16]:


figure = px.histogram(data, x = "Organic_carbon", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Organic Carbon")
figure.show()

#Organic carbon comes from the breakdown of natural organic materials and synthetic sources. Water containing less than 25 milligrams of organic carbon is considered safe to drink. 


# In[17]:


figure = px.histogram(data, x = "Trihalomethanes", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Trihalomethanes")
figure.show()

#THMs are chemicals found in chlorine-treated water. Water containing less than 80 milligrams of THMs is considered safe to drink. 


# In[18]:


plt.figure(figsize=(15,7))
plt.subplot(1,2,1)
plt.title("Almost Normal Distribution", fontsize=15)
sns.kdeplot(data = data["Sulfate"])


# In[19]:


sns.scatterplot(data=data, x='ph', y='Organic_carbon', hue='Potability')
plt.show()


# In[20]:


sns.scatterplot(data=data, x='ph', y='Chloramines', hue='Potability')
plt.show()


# In[21]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

# Load the Water Potability dataset
# Replace 'water_potability.csv' with the path to your dataset file
data = pd.read_csv('water_potability.csv')

# Handle missing values if any
data.fillna(data.mean(), inplace=True)

# Define features and target
X = data.drop('Potability', axis=1)
y = data['Potability']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test_scaled)
y_scores = clf.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Compute ROC curve and ROC AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = roc_auc_score(y_test, y_scores)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[22]:


correlation = data.corr()
correlation["Potability"].sort_values(ascending=False)


# In[23]:


# Visualizing dataset and also checking for outliers 

fig, ax = plt.subplots(ncols = 5, nrows = 2, figsize = (20, 10))
index = 0
ax = ax.flatten()

for col, value in data.items():
    sns.boxplot(y=col, data=data, ax=ax[index])
    index += 1
plt.tight_layout(pad = 0.5, w_pad=0.7, h_pad=5.0)


# In[67]:


sns.pairplot(data, hue="Potability")


# In[68]:


X = data.drop('Potability', axis=1)
y = data['Potability']


# In[69]:


X.shape, y.shape


# In[70]:


# import train-test split 
from sklearn.model_selection import train_test_split


# In[71]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.37, random_state=50)


# In[72]:


#Using Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[73]:


# Creating model object
model_lg = LogisticRegression(max_iter=120,random_state=42, n_jobs=20)


# In[74]:


# Training Model
model_lg.fit(X_train, y_train)


# In[75]:


# Making Prediction
pred_lg = model_lg.predict(X_test)


# In[76]:


# Calculating Accuracy Score
lg = accuracy_score(y_test, pred_lg)
print(lg)


# In[77]:


print(classification_report(y_test,pred_lg))


# In[78]:


# confusion Maxtrix
cm1 = confusion_matrix(y_test, pred_lg)
sns.heatmap(cm1/np.sum(cm1), annot = True, fmt=  '0.2%', cmap = 'Reds')


# In[79]:


#Using Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier


# In[80]:


# Creating model object
model_dt = DecisionTreeClassifier( max_depth=6, random_state=100)


# In[81]:


# Training Model
model_dt.fit(X_train,y_train)


# In[82]:


# Making Prediction
pred_dt = model_dt.predict(X_test)


# In[83]:


# Calculating Accuracy Score
dt = accuracy_score(y_test, pred_dt)
print(dt)


# In[84]:


print(classification_report(y_test,pred_dt))


# In[85]:


# confusion Maxtrix
cm2 = confusion_matrix(y_test, pred_dt)
sns.heatmap(cm2/np.sum(cm2), annot = True, fmt=  '0.2%', cmap = 'Reds')


# In[86]:


#Using Random Forest
from sklearn.ensemble import RandomForestClassifier


# In[87]:


# Creating model object
model_rf = RandomForestClassifier(n_estimators=300,min_samples_leaf=0.4, random_state=42)


# In[88]:


# Training Model
model_rf.fit(X_train, y_train)


# In[89]:


# Making Prediction
pred_rf = model_rf.predict(X_test)


# In[90]:


# Calculating Accuracy Score
rf = accuracy_score(y_test, pred_rf)
print(rf)


# In[91]:


print(classification_report(y_test,pred_rf))


# In[92]:


# confusion Maxtrix
cm3 = confusion_matrix(y_test, pred_rf)
sns.heatmap(cm3/np.sum(cm3), annot = True, fmt=  '0.2%', cmap = 'Reds')


# In[93]:


#Using KNeighbours
from sklearn.neighbors import KNeighborsClassifier


# In[94]:


# Creating model object
model_kn = KNeighborsClassifier(n_neighbors=9, leaf_size=20)


# In[95]:


# Training Model
model_kn.fit(X_train, y_train)


# In[96]:


# Making Prediction
pred_kn = model_kn.predict(X_test)


# In[97]:


# Calculating Accuracy Score
kn = accuracy_score(y_test, pred_kn)
print(kn)


# In[98]:


print(classification_report(y_test,pred_kn))


# In[99]:


# confusion Maxtrix
cm5 = confusion_matrix(y_test, pred_kn)
sns.heatmap(cm5/np.sum(cm5), annot = True, fmt=  '0.2%', cmap = 'Reds')


# In[100]:


#Using SVM
from sklearn.svm import SVC, LinearSVC


# In[101]:


model_svm = SVC(kernel='rbf', random_state = 42)


# In[102]:


model_svm.fit(X_train, y_train)


# In[103]:


# Making Prediction
pred_svm = model_svm.predict(X_test)


# In[104]:


# Calculating Accuracy Score
sv = accuracy_score(y_test, pred_svm)
print(sv)


# In[105]:


print(classification_report(y_test,pred_kn))


# In[106]:


# confusion Maxtrix
cm6 = confusion_matrix(y_test, pred_svm)
sns.heatmap(cm6/np.sum(cm6), annot = True, fmt=  '0.2%', cmap = 'Reds')


# In[107]:


models = pd.DataFrame({
    'Model':['Logistic Regression', 'Decision Tree', 'Random Forest','KNeighbours', 'SVM',],
    'Accuracy_score' :[lg, dt, rf, kn, sv,]
})
models
sns.barplot(x='Accuracy_score', y='Model', data=models)

models.sort_values(by='Accuracy_score', ascending=False)


# In[108]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load your dataset
data = pd.read_csv("water_potability.csv")

# Basic EDA
print(data.head(29000))
print(data.info())
print(data.describe())
print(data.isnull().sum())

# Handling missing values
data.fillna(data.mean(), inplace=True)

# Feature scaling
scaler = StandardScaler()
features = data.drop('Potability', axis=1)
target = data['Potability']
scaled_features = scaler.fit_transform(features)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.3, random_state=42,)

# Initialize and train the best model
best_model = DecisionTreeClassifier()
best_model.fit(X_train, y_train)

# Predictions on test set
y_pred = best_model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# New data for prediction
new_data = pd.DataFrame({
    'ph': [1.7],
    'Hardness': [0],
    'Solids': [1.9],
    'Chloramines': [3.7],
    'Sulfate': [5.7],
    'Conductivity': [5.9],
    'Organic_carbon': [1.8],
    'Trihalomethanes': [5.3],
    'Turbidity': [1]
    
    # Add all necessary features
})

# Preprocess the new data
scaled_new_data = scaler.transform(new_data)

# Make predictions with the best model
predictions = best_model.predict(scaled_new_data)
print('Predictions:', predictions)
# # Assuming 1 indicates 'safe' and 0 indicates 'unsafe'

prediction_result = 'unsafe' if predictions[0] == 0 else 'safe'
print('Prediction Result:', prediction_result)



# In[109]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load your dataset
data = pd.read_csv("water_potability.csv")

# Basic EDA
print(data.head(29000))
print(data.info())
print(data.describe())
print(data.isnull().sum())

# Handling missing values
data.fillna(data.mean(), inplace=True)

# Feature scaling
scaler = StandardScaler()
features = data.drop('Potability', axis=1)
target = data['Potability']
scaled_features = scaler.fit_transform(features)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.3, random_state=42,)

# Initialize and train the best model
best_model = DecisionTreeClassifier()
best_model.fit(X_train, y_train)

# Predictions on test set
y_pred = best_model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# New data for prediction
new_data = pd.DataFrame({
     'ph': [3.0],
    'Hardness': [50],
    'Solids': [5000],
    'Chloramines': [10.0],
    'Sulfate': [400],
    'Conductivity': [2000],
    'Organic_carbon': [30],
    'Trihalomethanes': [100],
    'Turbidity': [8]
    # Add all necessary features
})

# Preprocess the new data
scaled_new_data = scaler.transform(new_data)

# Make predictions with the best model
predictions = best_model.predict(scaled_new_data)
print('Predictions:', predictions)
# # Assuming 1 indicates 'safe' and 0 indicates 'unsafe'
prediction_result = 'unsafe' if predictions[0] == 0 else 'safe'
print('Prediction Result:', prediction_result)


# In[ ]:





# In[ ]:




