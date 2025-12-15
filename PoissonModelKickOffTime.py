import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson
from decimal import Decimal

max_goals = 6
decay_rate = 0.004

Data = pd.read_csv("Premier_League_All_Seasons.csv", skiprows=0, skipfooter=530, engine='python')
List = pd.read_csv("Premier_League_All_Seasons.csv", skiprows=range(0, 11944), skipfooter=50, engine="python")

def calculate_weights(dates, x):
    dates = pd.to_datetime(dates, errors="coerce")
    latest_date = dates.max()
    diff_half_weeks = ((latest_date - dates).dt.days)
    return np.exp(-x * diff_half_weeks)

def prediction_poisson(Data, HomeTeam, AwayTeam, x, Time_of_Match):
    home_df = pd.DataFrame(data={"team": Data.HomeTeam, "opponent": Data.AwayTeam, "goals": Data.FTHG, "home": 1,
                                 "Date": pd.to_datetime(Data.Date, errors="coerce"), "Time": Data.Time})
    away_df = pd.DataFrame(data={"team": Data.AwayTeam, "opponent": Data.HomeTeam, "goals": Data.FTAG, "home": 0,
                                 "Date": pd.to_datetime(Data.Date, errors="coerce"), "Time": Data.Time})
    full_df = pd.concat([home_df, away_df]).reset_index(drop=True)
    full_df['Time'] = full_df['Time'].fillna('Unknown')
    full_df['weights'] = calculate_weights(full_df['Date'], x)

    model_Poisson = smf.glm(data=full_df, family=sm.families.Poisson(),
                            formula="goals ~ home + team + opponent + Time", freq_weights=full_df['weights']).fit()

    home_goals = \
    (model_Poisson.predict(pd.DataFrame(data={"team": HomeTeam, "opponent": AwayTeam, "home": 1, "Time": Time_of_Match}, index=[1])).values[0])
    away_goals = \
    model_Poisson.predict(pd.DataFrame(data={"team": AwayTeam, "opponent": HomeTeam, "home": 0, "Time": Time_of_Match}, index=[1])).values[0]

    Probability_matrix = [[poisson.pmf(i, team_avg) for i in range(0, max_goals)]
                          for team_avg in [home_goals, away_goals]]

    matrix = np.outer(np.array(Probability_matrix[0]), np.array(Probability_matrix[1]))

    Home_Win = np.sum(np.tril(matrix, -1))
    Draw = np.sum(np.diag(matrix))
    Away_Win = np.sum(np.triu(matrix, 1))

    if Home_Win == max(Away_Win, Home_Win, Draw):
        return "H"
    elif Away_Win == max(Draw, Away_Win, Home_Win):
        return "A"
    else:
        return "D"

def compare_prediction_poisson_once(x):
    temp_poisson, matches = 0, 0
    for i in range(0, len(List)):
        Data = pd.read_csv("premier_league_all_seasons_cleaned_testfile.csv", skiprows=0, skipfooter=430 - i,
                               engine='python')
        if List.iloc[i, 8] in Data.Time:
            if prediction_poisson(Data, List.iloc[i, 1], List.iloc[i, 2], x, List.iloc[i, 8]) == List.iloc[i, 5]:
                temp_poisson += 1
        else:
            if prediction_poisson(Data, List.iloc[i, 1], List.iloc[i, 2], x, "Unknown") == List.iloc[i, 5]:
                temp_poisson += 1

    print(f"The Poisson Distribution got {(temp_poisson / len(List)) * 100} % correct")

if __name__ == "__main__":
    compare_prediction_poisson_once(decay_rate)

