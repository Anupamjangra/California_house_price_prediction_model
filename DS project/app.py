import streamlit as st
import pandas as pd
import random 
import pickle 
from sklearn.preprocessing import StandardScaler
# Title

col = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']

st.title('California Housing Price Prediction')

st.image('https://fox8.com/wp-content/uploads/sites/12/2014/09/mansion.gif')

st.header('A model of housing prices to predict median house value in California',divider=True)

#st.header('''User Must Enter Given Value to predict Price:
#[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup]''')

st.sidebar.title('Select House Features 🏠')


st.sidebar.image('https://cdna.artstation.com/p/assets/images/images/032/450/520/large/hamza-hanif-picsart-11-27-03-18-25.jpg?1606476838')

temp_df = pd.read_csv('california.csv')

random.seed(50)

all_values = []
for i in temp_df[col]:
    min_value, max_value = temp_df[i].agg([min,max])

    var =st.sidebar.slider(f'select {i} value', int(min_value), int(max_value), 
                      random.randint(int(min_value),int(max_value)))
    all_values.append(var)

ss = StandardScaler()
ss.fit(temp_df[col])

final_value = ss.transform([all_values])

with open('house_price_pred_ridge_model.pkl','rb') as f:
    chatgpt = pickle.load(f)

price = chatgpt.predict(final_value)[0]

import time
value = 0
st.write(pd.DataFrame(dict(zip(col,all_values)),index = [1]))   # ZIP:- It means that it combine two diff. values. 
progress_bar = st.progress(value)
placeholder = st.empty()
placeholder.subheader('Predicting Price')
place = st.empty()
place.image('https://cdn-icons-gif.flaticon.com/11677/11677497.gif',width = 100)
if price>0:
    for i in range(100):
        time.sleep(0.05)
        progress_bar.progress(i+1)
        
    body =f'Predicted median House Price: ${round(price,2)} Thousand Dollars'
    placeholder.empty()
    place.empty()
    st.success(body)
else:
    body = 'Invalid House features Value'
    st.warning(body)
    