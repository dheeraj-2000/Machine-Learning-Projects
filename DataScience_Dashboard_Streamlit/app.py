import streamlit as st
import pandas as pd
import numpy as np
from sodapy import Socrata
import pydeck as pdk
import plotly.express as px

client = Socrata("data.cityofnewyork.us", None)
DATA_URL = client.get("h9gi-nx95", limit=100)
# results_df = pd.DataFrame.from_records(DATA_URL)


st.title("Motor Vehicle Collisions in NYC")
st.markdown("You can analyse Motor Vehicle Collisions happening in New York City using this Interactive Dashboard. The Dataset used in this Dashboard is getting updated daily.")


def load_data():
    data = pd.DataFrame.from_records(DATA_URL)
    data.dropna(subset=['latitude', 'longitude'], inplace=True)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data.rename(columns={'crash_date': 'date/time'}, inplace=True)
    # data["number_of_persons_injured"] = pd.to_numeric(data["number_of_persons_injured"])
    # data["in"] = pd.to_numeric(data["number_of_persons_injured"])
    # data["number_of_persons_injured"].astype(int)
    data["latitude"] = data["latitude"].astype(float)
    data["longitude"] = data["longitude"].astype(float)
    data["number_of_persons_injured"] = data["number_of_persons_injured"].astype(int)
    data["number_of_pedestrians_injured"] = data["number_of_pedestrians_injured"].astype(int)
    data["number_of_cyclist_injured"] = data["number_of_cyclist_injured"].astype(int)
    data["number_of_motorist_injured"] = data["number_of_motorist_injured"].astype(int)
    data['crash_time']= pd.to_datetime(data['crash_time'])

    return data


data = load_data()
original_data = data

st.header("Where are the most people injured in NYC?")
injured_people =  st.slider("Select Number of injured people", 0, 19)
st.map(data.query("number_of_persons_injured >= @injured_people")[["latitude", "longitude"]].dropna(how=("any")))


st.header("How many Collisions occurs during a given time of day?")
hour = st.slider("Hour to look at", 0, 23)
data = data[data['crash_time'].dt.hour == hour]

st.markdown("Vehicle Collisions b/w %i:00 and %i:00" % (hour, (hour+1)))
midpoint = (np.mean(data["latitude"]), np.mean(data["longitude"]))

st.write(pdk.Deck(
    map_style = "mapbox://styles/mapbox/light-v9",
    initial_view_state={
        "latitude": midpoint[0],
        "longitude": midpoint[1],
        "zoom": 11,
        "pitch":50,
    },
    layers=[
        pdk.Layer(
        "HexagonLayer",
        data=data[['crash_time', 'latitude', 'longitude']],
        get_position=['longitude', 'latitude'],
        radius=100,
        extruded=True,
        pickable=True,
        elevation_scale=4,
        elevation_range = [0, 1000],
        ),
    ],
))

st.subheader("Breakdown by minute b/w %i:00 and %i:00" % (hour, (hour+1) %24))
filtered = data[
    (data['crash_time'].dt.hour >= hour) & (data['crash_time'].dt.hour < (hour+1))

]
hist = np.histogram(filtered['crash_time'].dt.minute, bins=60, range=(0,60))[0]
chart_data = pd.DataFrame({'minute': range(0,60), 'crashes':hist})
fig = px.bar(chart_data, x='minute', y='crashes', hover_data=['minute', 'crashes'], height=400)
st.write(fig)


st.header("Top 5 dangerous streets by sorted by affecting type")
select = st.selectbox('Affected type of people', ['Pedestrians', 'Cyclists', 'Motorists'])

if select == 'Pedestrians':
    st.write(original_data.query("number_of_pedestrians_injured >=1")[["on_street_name", "number_of_pedestrians_injured"]].sort_values(by=['number_of_pedestrians_injured'], ascending=False).dropna(how='any').head())


elif select == 'Cyclists':
    st.write(original_data.query("number_of_cyclist_injured >=1")[["on_street_name", "number_of_cyclist_injured"]].sort_values(by=['number_of_cyclist_injured'], ascending=False).dropna(how='any').head())


elif select == 'Motorists':
    st.write(original_data.query("number_of_motorist_injured >=1")[["on_street_name", "number_of_motorist_injured"]].sort_values(by=['number_of_motorist_injured'], ascending=False).dropna(how='any').head())


if st.checkbox("Show Raw Data", False):
    st.subheader('Raw Data')
    st.write(data)
