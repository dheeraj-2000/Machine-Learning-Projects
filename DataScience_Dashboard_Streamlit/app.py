import streamlit as st
import pandas as pd
import numpy as np
from sodapy import Socrata
import pydeck as pdk

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
    data['crash_time']= pd.to_datetime(data['crash_time'])

    return data


data = load_data()

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
))






if st.checkbox("Show Raw Data", False):
    st.subheader('Raw Data')
    st.write(data)
