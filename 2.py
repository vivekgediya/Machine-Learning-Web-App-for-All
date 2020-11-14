import streamlit as st 
st.set_option('deprecation.showPyplotGlobalUse', False)

import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection

from PIL import Image

st.title("Machine Learning Web Application for All")

image = Image.open('3.jpg')
st.image(image, use_color=True,use_column_width=True)

st.write("""## **A simple ML Web App using Streamlit(Python)**""")
st.write("""### ** **Let's Explore different types of classifier and datasets**""")


def main():
    st.sidebar.write('## Select appropriate activities from below')
    activities = ['EDA','Visualization','Model Selection']
    option = st.sidebar.selectbox('Select Option',activities)

    data = st.file_uploader('Upload Dataset',type=['csv', 'json','xlsx','txt'])
    if data is not None:
            data.seek(0)
            df = pd.read_csv(data)
            st.dataframe(df.head(50))
            st.success('Data uploaded & Loaded successfully')

    # EDA Operations
    if option=='EDA':
        st.subheader('Exploratory Data Analysis')
        
        if st.checkbox('Display Shape'):
            st.write(df.shape)
        if st.checkbox('Display columns'):
            st.write(df.columns)
        if st.checkbox('Display Summary'):
            st.write(df.describe().T)
        if st.checkbox('Display Null Values'):
            st.write(df.isnull().sum())
        if st.checkbox('Display Data Types'):
            st.write(df.dtypes)
        if st.checkbox('Display Correlations of Columns'):
                st.write(df.corr())

    # Visualization Methods
    elif option=='Visualization':
        st.subheader("Data Visualization")

        if st.checkbox('Select Multiple Columns to plot *'):
            selected_columns = st.multiselect('Select Columns',df.columns)
            df1 = df[selected_columns]
            st.dataframe(df1)

        if st.checkbox('Disply Heatmap'):
            st.warning('Please select multiple columns first to plot pairplot')
            st.write(sns.heatmap(df1.corr(),linewidths=.5,vmax=1,annot=True,cmap='viridis',square=True))
            st.pyplot()
        
        if st.checkbox('Display Pairplot'):
            st.warning('Please select multiple columns first to plot pairplot')
            st.write(sns.pairplot(df1,diag_kind='kde',))
            st.pyplot()

        if st.checkbox('Display Pie chart'):
            all_columns = df.columns.to_list()
            pie_columns = st.selectbox('Select columns to dispaly pie chart',all_columns)
            piechart = df[pie_columns].value_counts().plot.pie(autopct='%1.1f%%')
            st.write(piechart)
            st.pyplot()


    # Model Building
    elif option=='Model Selection':
        st.subheader('Model Selection : Choose appropiate model for your data')

        if st.checkbox('Select X and y from Columns'):
            
            # X and y variables
            all_columns = df.columns.to_list()
            target = st.selectbox('Select columns that is Depended or Output attribute called as "y"',all_columns)
            y = df[target]
            st.dataframe(y)
            st.write('Target variable y is:',target)

            input = st.multiselect('Select columns that is Independed or Input attribute called as "X"',all_columns)
            X = df[input]
            st.dataframe(X)
        
        seed = st.sidebar.slider('Seed',0,50,0,10)
        st.sidebar.write('#### Choose Algorithm from here')
        classifier_name = st.sidebar.selectbox('Select appropiate Classifier',('K-Nearest Neighbours','Support Vector Machine','Logistic Regression','Naive Bayes','Decision Tree'))
         
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

        def get_classifier(name_of_clf,params): 
            clf = None
            if name_of_clf == 'Support Vector Machine': 
                clf = SVC(C=params['C'])
            elif name_of_clf =='K-Nearest Neighbours': 
                clf = KNeighborsClassifier(n_neighbors=params['K'])
            elif name_of_clf == 'Logistic Regression':
                clf = LogisticRegression()
            elif name_of_clf == 'Naive Bayes':
                clf = GaussianNB()
            elif name_of_clf == 'Decision Tree':
                clf = DecisionTreeClassifier()
            else:
                st.warning('Unknown classifier or choose from above')
            return clf

        clf = get_classifier(classifier_name,params) 

        if st.checkbox('Split data into train and test'): 
            x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=seed)
            st.success('Done!')

        if st.checkbox('Fit model'):
            clf.fit(x_train,y_train) 
            st.success('Training Completed!')

        if st.checkbox('Show predicted output'):
            y_pred = clf.predict(x_test)
            st.write(y_pred)

        if st.checkbox('Accuracy of model'):
            accuracy = accuracy_score(y_test,y_pred)
            st.write('Accuracy of ',classifier_name,'classifier is ',accuracy)

    st.sidebar.write('');st.sidebar.write('');st.sidebar.write('');st.sidebar.write('');st.sidebar.write('');st.sidebar.write('');st.sidebar.write('');st.sidebar.write('');st.sidebar.write('');st.sidebar.write(''); 
    st.sidebar.write('## About')
    st.sidebar.write('This is interactive Web App for Machine learning.')
    st.sidebar.write('Visit - https://github.com/vivekgediya for more ML projects')
  

if __name__ == '__main__':
    main()























