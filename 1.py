import streamlit as st 
st.set_option('deprecation.showPyplotGlobalUse', False)

import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from PIL import Image

st.title("1 Machine Learning Web Application for All")

image = Image.open('3.jpg')
st.image(image, use_color=True,use_column_width=True)

st.write("""## **1 A simple ML Web App using Streamlit(Python)**""")
st.write("""### ** **Let's Explore different types of classifire and datasets**""")

st.sidebar.write("### Sidebar Menu to choose Dataset and Classifier ")

dataset_name = st.sidebar.selectbox('Select predefined Dataset',('Breast cancer','Iris','Wine'))
classifier_name = st.sidebar.selectbox('Select classifier',('Support Vector Machine','K-Nearest Neighbours'))

# Function to select Datasets
def get_dataset(name):
    data=None
    if name == 'Iris':
        data=datasets.load_iris()
    elif name == 'Wine':
        data=datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    
    x=data.data # Data of dataset
    y=data.target # Target variable of dataset

    return x,y

x,y=get_dataset(dataset_name)
st.dataframe(x)

st.write('Shape of Dataset ',x.shape)
st.write('Unique target variales: ',len(np.unique(y)))

st.write('### ** **Visualization**')

# Boxplot
st.write('#### (1) Boxplot')
sns.boxplot(data=x, orient='h')
st.pyplot()

# Histogram
st.write('#### (2) Histogram')
plt.hist(x)
st.pyplot()

# Building Algorithm

def add_parameter(name_of_clf):
    params = dict()
    if name_of_clf == 'K-Nearest Neighbours': 
        K = st.sidebar.slider('K',1,15)
        params['K'] = K

    elif name_of_clf == 'Support Vector Machine': 
        C = st.sidebar.slider('C',0.01,15.0)
        params['C'] = C 
        
    return params

params = add_parameter(classifier_name)

# Accessing Classifier

def get_classifier(name_of_clf,params): 
    clf = None
    if name_of_clf == 'Support Vector Machine': 
        clf = SVC(C=params['C'])
    else: 
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    return clf

clf = get_classifier(classifier_name,params)    

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)

st.write('### ** **Result**')
st.write('Accuracy of ',classifier_name,'classifier is ',accuracy)















