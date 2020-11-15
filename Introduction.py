import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.figure_factory as ff
from PIL import Image
import time


st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Introduction to Streamlit")

st.subheader("Subheader")

image = Image.open('0.jpg')
st.image(image, use_color=True)

st.write('Write a Text here')

st.markdown('Markdown here')

st.success('Congratulations')

st.info('This is Info')

st.warning('This is Warning')

st.error('This is Error')

dataframe = np.random.rand(50, 20)
st.dataframe(dataframe)

st.text('****'*22)

df = pd.DataFrame(np.random.rand(10, 20), columns=(
    'col %d' % i for i in range(20)))
st.dataframe(df.style.highlight_max(axis=1))

st.text('****'*22)

chart_data = pd.DataFrame(np.random.rand(10,3), columns=['a','b','c'])
st.line_chart(chart_data)

st.text('****'*22)

st.area_chart(chart_data)

st.text('****'*22)

chart_data = pd.DataFrame(np.random.rand(50,3), columns=['a','b','c'])
st.bar_chart(chart_data)

arr = np.random.normal(1,1,size=100)
plt.hist(arr,bins=20)
st.pyplot()

x1=np.random.randn(200)-2
x2=np.random.randn(200)
x3=np.random.randn(200)-2

hist_data=[x1,x2,x3]
group_label=['group 1', 'group 2', 'group 3']
fig = ff.create_distplot(hist_data,group_label,bin_size=[0.2,0.25,0.5])
st.plotly_chart(fig,use_container_width=True)

df = pd.DataFrame(np.random.randn(100,2)/(50,50)*(37.76,-122.4), columns=['lat', 'lon'])
st.map(df)

if st.button("A"):
    st.write('S')
else: 
    st.write('Go')   


genere = st.radio('gender',('male','female'))
if genere == 'male':
    st.write('Male')
elif genere == 'female':
    st.write('Female')

option = st.selectbox('age',('<20','20-50','>50'))
st.write('Age', option)

option = st.multiselect('age',('<20','20-50','>50'))
st.write('Age', option)

age = st.slider('age is',0,100,1)
st.write('Age is',age)


age1111 = st.slider('age is',0,100,(1,20))
st.write('Ageeeeee is',age1111)

nubmer = st.number_input('Input Number')
st.write('number is',nubmer)

# upload_file = st.file_uploader('Choose a csv file')

# if upload_file is not None:
#     data = pd.read_csv(upload_file)
#     st.write('Data',data)
#     st.success('Congratulations Uploaded!')
# else:
#     st.error('Invalid upload')

color = st.sidebar.color_picker("Pic color: ",'#00f900')
st.sidebar.write('This is', color)

add_sidebar = st.sidebar.selectbox('hello!',('hi','by','khelo','zelo','melo'))

my_bar = st.progress(0)
for percent_complete in range(100): 
    time.sleep(0.1)
    my_bar.progress(percent_complete+1)

with st.spinner('wait for..'):
    time.sleep(5)
st.success("Oh! yeah")
st.balloons()