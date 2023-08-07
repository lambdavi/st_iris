import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import numpy as np

st.write("""
    # Iris dataset classification
    """)

st.sidebar.header('User input')

def user_input_features():
    sepal_length = st.sidebar.slider("Sepal Length", 4.3, 7.9, 5.3)
    sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider("Petal Length", 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
st.subheader("User Input Features")
st.write(df)

iris = load_iris()
X = iris.data
y = iris.target

st.write("""
    ### Classifier: Random Forest, example of train dataset below:
        """)

st.write(np.hstack((X[:5, :], iris.target_names[y[:5].reshape(-1,1)])))
st.write("""
    ### Possible values of target variable:
        """)
st.write(iris.target_names)
model = RandomForestClassifier()
model.fit(X, y)

prediction = model.predict(df)

st.subheader("Prediction based on your input")
st.write(iris.target_names[prediction])

