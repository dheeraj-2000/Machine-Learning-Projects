import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Interaction area")
    st.markdown("are your mushrooms edible or poisonous? 🍄")
    st.sidebar.markdown("are your mushrooms edible or poisonous? 🍄")

    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv('/home/dheeraj/my_projects/my_project_env/practice/motion_detector/ML_Dashboard_Streamlit/mushrooms.csv')
        label = LabelEncoder()
        for each_col in data.columns:
            data[each_col] = label.fit_transform(data[each_col])
        return data

    df = load_data()

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom dataset for Classification")
        st.write(df)



if __name__ == '__main__':
    main()
