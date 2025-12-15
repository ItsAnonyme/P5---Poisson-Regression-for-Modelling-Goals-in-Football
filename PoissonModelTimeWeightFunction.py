import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson
from collections import Counter

max_goals = 6
decay_rate = 0.004
accuracy_bound = 0.8
lower_bound_sum = 0
lower_bound_individual = 0

Data = pd.read_csv("Premier_League_All_Seasons.csv", skiprows=0, skipfooter=150, engine='python')
List = pd.read_csv("Premier_League_All_Seasons.csv", skiprows=range(0, 12324), skipfooter=0, engine="python")

home_df = pd.DataFrame(
    data={"team": Data.HomeTeam, "opponent": Data.AwayTeam, "goals": Data.FTHG, "home": 1, "Date": Data.Date})
away_df = pd.DataFrame(
    data={"team": Data.AwayTeam, "opponent": Data.HomeTeam, "goals": Data.FTAG, "home": 0, "Date": Data.Date})
full_df = pd.concat([home_df, away_df]).reset_index(drop=True)

def calculate_weights(dates, x):
    dates = pd.to_datetime(dates, errors='coerce')
    latest_date = dates.max()
    diff_half_weeks = ((latest_date - dates).dt.days)
    weights = np.exp(-x * diff_half_weeks)
    return weights

def model(x):
    full_df['weights'] = calculate_weights(full_df['Date'], x)
    model_Poisson = smf.glm(data=full_df, family=sm.families.Poisson(),
                            formula="goals ~ home + team + opponent", freq_weights=full_df['weights']).fit()
    return model_Poisson

model_test = model(decay_rate)

def prediction_poisson(HomeTeam, AwayTeam, poisson_model):
    home_goals = \
    (poisson_model.predict(pd.DataFrame(data={"team": HomeTeam, "opponent": AwayTeam, "home": 1}, index=[1])).values[0])
    away_goals = \
    poisson_model.predict(pd.DataFrame(data={"team": AwayTeam, "opponent": HomeTeam, "home": 0}, index=[1])).values[0]

    probs = np.outer(poisson.pmf(range(max_goals), home_goals),
                     poisson.pmf(range(max_goals), away_goals))

    max_index = np.argmax(probs)
    i, j = np.unravel_index(max_index, probs.shape)

    Home_Win = np.sum(np.tril(probs, -1))
    Draw = np.sum(np.diag(probs))
    Away_Win = np.sum(np.triu(probs, 1))

    if Home_Win > Away_Win and Home_Win > Draw:
        return "H", Home_Win, Away_Win, Draw, Home_Win, i, j
    elif Away_Win > Draw and Away_Win > Home_Win:
        return "A", Home_Win, Away_Win, Draw, Away_Win, i, j
    else:
        return "D", Home_Win, Away_Win, Draw, Draw, i, j

def compare_prediction_poisson_once():
        poisson_result, poisson_score, acc_bound_match, acc_result_win, acc_score_win = 0, 0, 0, 0, 0
        brier = 0
        result, score_result, actual_result = Counter(), Counter(), Counter()
        for i in range(len(List)):
            prediction, home_prob, away_prob, draw_prob, win_prob, home_goals, away_goals = prediction_poisson(List.iloc[i, 1], List.iloc[i, 2], model_test)
            result[prediction] += 1
            actual_result[List.iloc[i, 5]] += 1
            if prediction == List.iloc[i, 5]:
                poisson_result += 1
            if win_prob > accuracy_bound:
                acc_bound_match += 1
                if prediction == List.iloc[i, 5]:
                    acc_result_win += 1
                if home_goals == List.iloc[i, 3] and away_goals == List.iloc[i, 4]:
                    acc_score_win += 1
                match List.iloc[i, 5]:
                    case "H":
                        brier += (1 - home_prob) ** 2 + (away_prob) ** 2 + (draw_prob) ** 2
                    case "A":
                        brier += (home_prob) ** 2 + (1 - away_prob) ** 2 + (draw_prob) ** 2
                    case _:
                        brier += (home_prob) ** 2 + (away_prob) ** 2 + (1 - draw_prob) ** 2
            if ((Data.HomeTeam == List.iloc[i, 1]).sum() + (Data.AwayTeam == List.iloc[i, 1]).sum()
            + (Data.HomeTeam == List.iloc[i, 2]).sum() + (Data.AwayTeam == List.iloc[i, 2]).sum() <= lower_bound_sum
            or (Data.HomeTeam == List.iloc[i, 1]).sum() + (Data.AwayTeam == List.iloc[i, 1]).sum() <= lower_bound_individual
            or (Data.HomeTeam == List.iloc[i, 2]).sum() + (Data.AwayTeam == List.iloc[i, 2]).sum() <= lower_bound_individual):
                continue
            else:
                continue
        print(f"Brier score: {brier/acc_bound_match} for {acc_bound_match} matches, with probability {(acc_result_win/acc_bound_match)*100} and {(acc_score_win/acc_bound_match)*100}")
        print(f"The Poisson Distribution got {(poisson_result/len(List)) * 100} % of the results correct")
        print(f"The Poisson Distribution got {(poisson_score/len(List)) * 100} % of the scores correct")
        print(dict(result))
        print(dict(score_result))
        print(dict(actual_result))


if __name__ == "__main__":
    compare_prediction_poisson_once()
