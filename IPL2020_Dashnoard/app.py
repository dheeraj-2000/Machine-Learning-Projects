import streamlit as st
import pandas as pd
import numpy as np
from sodapy import Socrata
import pydeck as pdk
import plotly.express as px
import requests

points_table_data_url = 'https://www.iplt20.com/points-table/2020'
# most_run_data_url = 'https://www.iplt20.com/stats/2020/most-runs'
html = requests.get(points_table_data_url).content
df_list_points_table = pd.read_html(html)
df_points_table = df_list_points_table[-1]
# print(df)
# df = pd.DataFrame(df)


st.title("IPL 2020 Dashboard")
st.markdown("You can check latest Status of **IPL 2020** along with various other stats of your favourite team 🏏")

@st.cache(persist=True)
def load_data_point_table():
    data = pd.DataFrame(df_points_table)
    data.rename(columns={'Unnamed: 0': 'Position', 'Pld': 'Match Played', 'Pts': 'Points', 'Form':'Status of Past Matches'}, inplace=True)
    data = data.replace(np.nan, 'Not Played yet')
    return data


data_point_table = load_data_point_table()
st.header("Points Table of IPL 2020")
st.write(data_point_table)

# Batting & Bowling stats of all team
batting_stats_data_url = 'https://www.iplt20.com/stats/2020/most-runs'
# most_run_data_url = 'https://www.iplt20.com/stats/2020/most-runs'
html = requests.get(batting_stats_data_url).content
df_list_batting_stat = pd.read_html(html)
df_batting_stat = df_list_batting_stat[-1]
# print(df)



st.header("Check Top Performers of Ongoing IPL Season")
select_bat_bowl = st.selectbox('Which stats you want to check?', ['--Select--', 'Batting stats', 'Bowling stats'])

if select_bat_bowl == 'Batting stats':

    @st.cache(persist=True)
    def load_data_batting_table():
        data = pd.DataFrame(df_batting_stat)
        data.rename(columns={'POS':'Position.',	'PLAYER': 'Player',	'Mat': 'Matches','Inns': 'Innings',	'NO':'Not Outs','HS': 'Highest Score',
        	                           'Avg': 'Average',	'BF': 'Ball Faced',	'SR': 'Strike Rate'	}, inplace=True)
        # data = data.replace(np.nan, 'Not Played yet')
        return data

    data_batting_stats = load_data_batting_table()

    if st.checkbox("Show Top 30 Batsman List (in terms of Total Runs Scored)", False):
        st.header("Batting Stats of top Players")
        st.write(data_batting_stats.head(30))

    st.subheader("Check Top 3 Batsman in Selective categories")
    select_category = st.selectbox('Choose the Performance category', ['--Select--', 'Top Run Scorer', 'Highest Strike Rate', 'Best Average'])

    if select_category == 'Top Run Scorer':
        df_bat_total_score = data_batting_stats.sort_values(by=['Runs'], ascending=False).head(3)
        x = np.arange(1, 4)
        df_bat_total_score['Position'] = x

        data_batting_stats_new = df_bat_total_score[['Position', 'Player', 'Runs']].head(3)

        fig = px.bar(data_batting_stats_new, x='Player', y='Runs', hover_data=['Position','Player', 'Runs'], color='Player')

        fig.update_layout(xaxis_title="Batsman",
                            yaxis_title="Total Runs Scored",
                            legend_title="Players",
                            font=dict(
                                family="Arial",
                                size=18,
                                color="RebeccaPurple"
                            ))

        fig.update_layout(title={'text': "Top 3 Most run scorer Batsman",
                                    'y':0.95,
                                    'x':0.45,
                                    'xanchor': 'center',
                                    'yanchor': 'top'})

        st.write(fig)

    elif select_category == 'Highest Strike Rate':
        df_bat_sr = data_batting_stats.sort_values(by=['Strike Rate'], ascending=False).head(3)
        x = np.arange(1, 4)
        df_bat_sr['Position'] = x
        data_batting_stats_sr = df_bat_sr[['Position', 'Player', 'Strike Rate']].head(3)

        fig2 = px.bar(data_batting_stats_sr, x='Player', y='Strike Rate', hover_data=['Position','Player', 'Strike Rate'], color='Player')

        fig2.update_layout(xaxis_title="Batsman",
                            yaxis_title="Strike Rate",
                            legend_title="Players",
                            font=dict(
                                family="Arial",
                                size=18,
                                color="RebeccaPurple"
                            ))

        fig2.update_layout(title={'text': "Top 3 Batsman with Highest Strike Rate)",
                                    'y':0.95,
                                    'x':0.45,
                                    'xanchor': 'center',
                                    'yanchor': 'top'})

        st.write(fig2)

    elif select_category == 'Best Average':
        best_avg_data_url = 'https://www.iplt20.com/stats/2020/best-batting-average'
        # most_run_data_url = 'https://www.iplt20.com/stats/2020/most-runs'
        html = requests.get(best_avg_data_url).content
        df_list_avg = pd.read_html(html)
        df_batting_stat_best_avg = df_list_avg[-1]

        @st.cache(persist=True)
        def load_data_batting_table():
            data = pd.DataFrame(df_batting_stat_best_avg)
            data.rename(columns={'PLAYER':'Player', 'Avg': 'Average'}, inplace=True)
            # data = data.replace(np.nan, 'Not Played yet')
            return data

        data_best_avg_stats = load_data_batting_table()


        df_bat_bestavg = data_best_avg_stats.sort_values(by=['Average'], ascending=False).head(3)
        x = np.arange(1, 4)
        df_bat_bestavg['Position'] = x
        data_batting_stats_bestavg = df_bat_bestavg[['Position', 'Player', 'Average']].head(3)

        fig2 = px.bar(data_batting_stats_bestavg, x='Player', y='Average', hover_data=['Position','Player', 'Average'], color='Player')

        fig2.update_layout(xaxis_title="Batsman",
                            yaxis_title="Best Average",
                            legend_title="Players",
                            font=dict(
                                family="Arial",
                                size=16,
                                color="RebeccaPurple"
                            ))

        fig2.update_layout(title={'text': "Top 3 Batsman with Best Average",
                                    'y':0.95,
                                    'x':0.42,
                                    'xanchor': 'center',
                                    'yanchor': 'top'})

        st.write(fig2)
