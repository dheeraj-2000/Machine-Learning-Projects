# https://fierce-brook-58056.herokuapp.com/
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
    st.title("Classification of Edible and Poisonous Mushrooms 🍄")
    st.sidebar.title("Interaction area")
    st.markdown("Use ML algorithms to check whether mushrooms are edible or poisonous? 🍄")
    st.sidebar.markdown("Select the Model and their Hyperparameters, and click on Classify button to see the results.")

    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv('mushrooms.csv')
        label = LabelEncoder()
        for each_col in data.columns:
            data[each_col] = label.fit_transform(data[each_col])
        return data

    @st.cache(persist=True)
    def split(df):
        x = df.drop(columns=['class'])
        y = df['class']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list):
        if 'Confusion Metrix' in metrics_list:
            st.subheader("Confusion Metrix")
            plot_confusion_matrix(model, x_test, y_test, display_labels = class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()

    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    df = load_data()
    class_names = ['edible', 'poisonous']

    st.sidebar.subheader("View Dataset")
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom dataset for Classification")
        st.write(df)


    st.sidebar.subheader("Choose Classifier from the Dropdown option")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest Classifier"))

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.1, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

        metrix = st.sidebar.multiselect("Select the metrics you want to plot (You can choose more than one)", ('Confusion Metrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy", accuracy.round(2))
            st.write("Precision", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrix)


    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.1, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iteration", 100, 500, key='max_iter')

        metrix = st.sidebar.multiselect("Select the metrics you want to plot.", ('Confusion Metrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy", accuracy.round(2))
            st.write("Precision", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrix)

    if classifier == 'Random Forest Classifier':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("Number of Decision Trees to be used", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("Depth of the Trees that will be used in the Forest", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples while building trees", ('True', 'False'), key='bootstrap')


        metrix = st.sidebar.multiselect("Select the metrics you want to plot.", ('Confusion Metrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Classifier Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy", accuracy.round(2))
            st.write("Precision", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrix)










if __name__ == '__main__':
    main()
