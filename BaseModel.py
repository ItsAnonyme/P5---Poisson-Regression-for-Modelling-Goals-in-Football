import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson
from decimal import Decimal

max_goals = 6

Data = pd.read_csv("Premier_League_All_Seasons.csv", skiprows=0, skipfooter=430, engine='python')
List = pd.read_csv("Premier_League_All_Seasons.csv", skiprows=range(0, 11944), skipfooter=50, engine="python")

home_df = pd.DataFrame(
    data={"team": Data.HomeTeam, "opponent": Data.AwayTeam, "goals": Data.FTHG, "home": 1})
away_df = pd.DataFrame(
    data={"team": Data.AwayTeam, "opponent": Data.HomeTeam, "goals": Data.FTAG, "home": 0})

def prediction_poisson(HomeTeam, AwayTeam):
    full_df = pd.concat([home_df, away_df]).reset_index(drop=True)

    model_Poisson = smf.glm(data=full_df, family=sm.families.Poisson(),
                            formula="goals ~ home + team + opponent").fit()

    home_goals = \
    (model_Poisson.predict(pd.DataFrame(data={"team": HomeTeam, "opponent": AwayTeam, "home": 1}, index=[1])).values[0])
    away_goals = \
    model_Poisson.predict(pd.DataFrame(data={"team": AwayTeam, "opponent": HomeTeam, "home": 0}, index=[1])).values[0]

    Probability_matrix = [[poisson.pmf(i, team_avg) for i in range(0, max_goals)]
                          for team_avg in [home_goals, away_goals]]

    matrix = np.outer(np.array(Probability_matrix[0]), np.array(Probability_matrix[1]))

    Home_Win = np.sum(np.tril(matrix, -1))
    Draw = np.sum(np.diag(matrix))
    Away_Win = np.sum(np.triu(matrix, 1))
    if Home_Win > Away_Win and Home_Win > Draw:
        return "H"
    elif Away_Win > Draw and Away_Win > Home_Win:
        return "A"
    else:
        return "D"

def compare_prediction_poisson():
    temp_poisson = 0
    for i in range(0, len(List)):
        if prediction_poisson(List.iloc[i, 1], List.iloc[i, 2]) == List.iloc[i, 5]:
            temp_poisson += 1
    print(f"The Poisson Distribution got {round(Decimal((temp_poisson/len(List)) * 100), 10)} % correct")

if __name__ == "__main__":
    compare_prediction_poisson()
