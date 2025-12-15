import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson
from collections import Counter
import math

max_goals = 6
decay_rate = 0.004
lower_bound_sum = 0
lower_bound_individual = 750
accuracy_bound = 0.6

Data = pd.read_csv("premier_league_all_seasons_cleaned_test.csv")

Data["Date"] = pd.to_datetime(Data["Date"], errors="coerce")
Data = Data.sort_values("Date")

home = Data[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].rename(
    columns={"HomeTeam": "team", "AwayTeam": "opponent", "FTHG": "goals_home", "FTAG": "goals_away"}
)
away = Data[["Date", "AwayTeam", "HomeTeam", "FTAG", "FTHG"]].rename(
    columns={"AwayTeam": "team", "HomeTeam": "opponent", "FTAG": "goals_home", "FTHG": "goals_away"}
)
all_matches = pd.concat([home, away]).sort_values(["team", "Date"])

all_matches["goals_last5"] = all_matches.groupby("team")["goals_home"].rolling(5, min_periods=1).mean().shift(
    1).reset_index(0, drop=True)
all_matches["conceded_last5"] = all_matches.groupby("team")["goals_away"].rolling(5, min_periods=1).mean().shift(
    1).reset_index(0, drop=True)

all_matches["last_result"] = np.where(
    all_matches["goals_home"] > all_matches["goals_away"], 1,
    np.where(all_matches["goals_home"] < all_matches["goals_away"], -1, 0)
)
all_matches["last_result_shifted"] = all_matches.groupby("team")["last_result"].shift(1)

Data = Data.merge(
    all_matches[["team", "Date", "goals_last5", "conceded_last5", "last_result_shifted"]],
    left_on=["HomeTeam", "Date"], right_on=["team", "Date"], how="left"
).rename(columns={
    "goals_last5": "team_goals_last5",
    "conceded_last5": "team_conceded_last5",
    "last_result_shifted": "team_last_result"
}).drop(columns=["team"])

Data = Data.merge(
    all_matches[["team", "Date", "goals_last5", "conceded_last5", "last_result_shifted"]],
    left_on=["AwayTeam", "Date"], right_on=["team", "Date"], how="left"
).rename(columns={
    "goals_last5": "opp_goals_last5",
    "conceded_last5": "opp_conceded_last5",
    "last_result_shifted": "opp_last_result"
}).drop(columns=["team"])

Data["pair"] = Data.apply(
    lambda row: "-".join(sorted([row["HomeTeam"], row["AwayTeam"]])),
    axis=1
)
Data["h2h_last"] = Data.groupby("pair")["FTR"].shift(1)
Data["h2h_last"] = Data["h2h_last"].map({'H': 1, 'A': -1, 'D': 0})

all_matches["last_match_date"] = all_matches.groupby("team")["Date"].shift(1)

Data = Data.merge(all_matches[["team", "Date", "last_match_date"]],
                  left_on=["HomeTeam", "Date"], right_on=["team", "Date"], how="left")
Data = Data.rename(columns={"last_match_date": "home_last_match"})

Data = Data.merge(all_matches[["team", "Date", "last_match_date"]],
                  left_on=["AwayTeam", "Date"], right_on=["team", "Date"], how="left")
Data = Data.rename(columns={"last_match_date": "away_last_match"})

Data["rest_days_home"] = (Data["Date"] - Data["home_last_match"]).dt.days
Data["rest_days_away"] = (Data["Date"] - Data["away_last_match"]).dt.days

Data["rest_days_home"] = Data["rest_days_home"].fillna(Data["rest_days_home"].median())
Data["rest_days_away"] = Data["rest_days_away"].fillna(Data["rest_days_away"].median())

home_df = pd.DataFrame({
    "team": Data["HomeTeam"],
    "opponent": Data["AwayTeam"],
    "goals": Data["FTHG"],
    "home": 1,
    "Date": Data["Date"],
    "Time": Data["Time"],
    "team_goals_last5": Data["team_goals_last5"],
    "team_conceded_last5": Data["team_conceded_last5"],
    "opp_goals_last5": Data["opp_goals_last5"],
    "opp_conceded_last5": Data["opp_conceded_last5"],
    "team_last_result": Data["team_last_result"],
    "opp_last_result": Data["opp_last_result"],
    "h2h_last": Data["h2h_last"],
    "team_rest_days": Data["rest_days_home"],
    "opp_rest_days": Data["rest_days_away"],
    "result": Data["FTR"]
})

away_df = pd.DataFrame({
    "team": Data["AwayTeam"],
    "opponent": Data["HomeTeam"],
    "goals": Data["FTAG"],
    "home": 0,
    "Date": Data["Date"],
    "Time": Data["Time"],
    "team_goals_last5": Data["opp_goals_last5"],
    "team_conceded_last5": Data["opp_conceded_last5"],
    "opp_goals_last5": Data["team_goals_last5"],
    "opp_conceded_last5": Data["team_conceded_last5"],
    "team_last_result": Data["opp_last_result"],
    "opp_last_result": Data["team_last_result"],
    "h2h_last": -Data["h2h_last"],
    "team_rest_days": Data["rest_days_away"],
    "opp_rest_days": Data["rest_days_home"],
    "result": Data["FTR"]
})
full_df = pd.concat([home_df, away_df]).reset_index(drop=True)
full_df['Time'] = full_df['Time'].fillna('Unknown')
full_df['team_last_result'] = full_df['team_last_result'].fillna('Unknown')
full_df['opp_last_result'] = full_df['opp_last_result'].fillna('Unknown')
full_df['h2h_last'] = full_df['h2h_last'].fillna('Unknown')

formula = ("goals ~ home + team + opponent "
           "+ h2h_last + team_goals_last5 + team_conceded_last5 "
           "+ opp_goals_last5 + opp_conceded_last5 "
           "+ team_last_result + opp_last_result "
           "+ team_rest_days + opp_rest_days")

def calculate_weights(dates, x):
    dates = pd.to_datetime(dates, errors='coerce')
    latest_date = dates.max()
    diff_half_weeks = ((latest_date - dates).dt.days)
    weights = np.exp(-x * diff_half_weeks)
    return weights

def prediction_poisson(i, time_of_match):
    rows = np.r_[0:i, 12474:12474 + i]
    df_train = full_df.iloc[rows, :].copy()
    df_train["weights"] = calculate_weights(df_train['Date'], decay_rate)
    model = smf.glm(data=df_train, family=sm.families.Poisson(),
                    formula=formula,
                    freq_weights=df_train['weights']).fit()

    Home_team = full_df['team'].iloc[i]
    Away_team = full_df['opponent'].iloc[i]
    tg_last5 = full_df['team_goals_last5'].iloc[i]
    tc_last5 = full_df['team_conceded_last5'].iloc[i]
    og_last5 = full_df['opp_goals_last5'].iloc[i]
    oc_last5 = full_df['opp_conceded_last5'].iloc[i]
    tlr = full_df['team_last_result'].iloc[i]
    olr = full_df['opp_last_result'].iloc[i]
    h2h_last_team = full_df['h2h_last'].iloc[i]
    h2h_last_opp = full_df['h2h_last'].iloc[12474 + i]
    team_rest_days = full_df['team_rest_days'].iloc[i]
    opp_rest_days = full_df['opp_rest_days'].iloc[i]

    home_goals = (model.predict(pd.DataFrame(data={"team": Home_team, "opponent": Away_team, "home": 1,
                                                   "team_goals_last5": tg_last5, "team_conceded_last5": tc_last5,
                                                   "opp_goals_last5": og_last5, "opp_conceded_last5": oc_last5,
                                                   "team_last_result": tlr, "opp_last_result": olr,
                                                   "h2h_last": h2h_last_team, "team_rest_days": team_rest_days,
                                                   "opp_rest_days": opp_rest_days}, index=[1])).values[0])

    away_goals = (model.predict(pd.DataFrame(data={"team": Away_team, "opponent": Home_team, "home": 0,
                                                   "team_goals_last5": og_last5, "team_conceded_last5": oc_last5,
                                                   "opp_goals_last5": tg_last5, "opp_conceded_last5": tc_last5,
                                                   "team_last_result": olr, "opp_last_result": tlr,
                                                   "h2h_last": h2h_last_opp, "team_rest_days": opp_rest_days,
                                                   "opp_rest_days": team_rest_days}, index=[1])).values[0])

    probs = np.outer(poisson.pmf(range(max_goals), home_goals),
                     poisson.pmf(range(max_goals), away_goals))

    max_index = np.argmax(probs)
    h, a = np.unravel_index(max_index, probs.shape)

    Home_Win = np.sum(np.tril(probs, -1))
    Draw = np.sum(np.diag(probs))
    Away_Win = np.sum(np.triu(probs, 1))

    if Home_Win > Away_Win and Home_Win > Draw:
        return "H", h, a, Home_Win, home_goals, away_goals
    elif Away_Win > Draw and Away_Win > Home_Win:
        return "A", h, a, Away_Win, home_goals, away_goals
    else:
        return "D", h, a, Draw, home_goals, away_goals

def compare_prediction_poisson_once():
        poisson_result_lower, poisson_result_upper, poisson_score_lower, poisson_score_upper, matches_lower, matches_upper = 0, 0, 0, 0, 0, 0
        error1_home_l, error1_away_l, error2_home_l, error2_away_l, deviance_home_l, deviance_away_l = 0, 0, 0, 0, 0, 0
        error1_home_u, error1_away_u, error2_home_u, error2_away_u, deviance_home_u, deviance_away_u = 0, 0, 0, 0, 0, 0
        match_accuracy, result_accuracy, score_accuracy, deviance_home_acc, deviance_away_acc = 0, 0, 0, 0, 0
        result, score_result, actual_result = Counter(), Counter(), Counter()
        for i in range(11944, 12324):
            rows = np.r_[0:i, 12474:12474 + i]
            time_of_match = full_df['Time'].iloc[i]
            if time_of_match in full_df.Time.iloc[rows]:
                prediction, home_goals, away_goals, winner, home_goals_predict, away_goals_predict = prediction_poisson(i, time_of_match)
                if winner > accuracy_bound:
                    match_accuracy += 1
                    if full_df['goals'].iloc[i] > 0:
                        deviance_home_acc += full_df['goals'].iloc[i] * math.log(full_df['goals'].iloc[i] / home_goals_predict) - (
                                full_df['goals'].iloc[i] - home_goals_predict)
                    else:
                        deviance_home_acc += - (full_df['goals'].iloc[i] - home_goals_predict)
                    if full_df['goals'].iloc[12474 + i] > 0:
                        deviance_away_acc += full_df['goals'].iloc[12474 + i] * math.log(
                            full_df['goals'].iloc[12474 + i] / away_goals_predict) - (
                                                   full_df['goals'].iloc[12474 + i] - away_goals_predict)
                    else:
                        deviance_away_acc += - (full_df['goals'].iloc[12474 + i] - away_goals_predict)
                    if prediction == full_df['result'].iloc[i]:
                        result_accuracy += 1
                    if home_goals == full_df['goals'].iloc[i] and away_goals == full_df['goals'].iloc[12474 + i]:
                        score_accuracy += 1
                if ((full_df.team.iloc[rows] == full_df['team'].iloc[i]).sum() + (full_df.team.iloc[rows] == full_df['opponent'].iloc[i]).sum() <= lower_bound_sum
                        or (full_df.team.iloc[rows] == full_df['team'].iloc[i]).sum() <= lower_bound_individual
                        or (full_df.team.iloc[rows] == full_df['opponent'].iloc[i]).sum() <= lower_bound_individual):
                    matches_lower += 1
                    if prediction == full_df['result'].iloc[i]:
                        poisson_result_lower += 1
                    if home_goals == full_df['goals'].iloc[i] and away_goals == full_df['goals'].iloc[12474 + i]:
                        poisson_score_lower += 1
                    error1_home_l += (home_goals_predict - full_df['goals'].iloc[i]) ** 2
                    error1_away_l += (home_goals - full_df['goals'].iloc[i]) ** 2
                    error2_home_l += (away_goals_predict - full_df['goals'].iloc[12474 + i]) ** 2
                    error2_away_l += (away_goals - full_df['goals'].iloc[12474 + i]) ** 2
                    if full_df['goals'].iloc[i] > 0:
                        deviance_home_l += full_df['goals'].iloc[i] * math.log(full_df['goals'].iloc[i] / home_goals_predict) - (
                                full_df['goals'].iloc[i] - home_goals_predict)
                    else:
                        deviance_home_l += - (full_df['goals'].iloc[i] - home_goals_predict)
                    if full_df['goals'].iloc[12474 + i] > 0:
                        deviance_away_l += full_df['goals'].iloc[12474 + i] * math.log(
                            full_df['goals'].iloc[12474 + i] / away_goals_predict) - (
                                                   full_df['goals'].iloc[12474 + i] - away_goals_predict)
                    else:
                        deviance_away_l += - (full_df['goals'].iloc[12474 + i] - away_goals_predict)
                else:
                    error1_home_u += (home_goals_predict - full_df['goals'].iloc[i]) ** 2
                    error1_away_u += (home_goals - full_df['goals'].iloc[i]) ** 2
                    error2_home_u += (away_goals_predict - full_df['goals'].iloc[12474 + i]) ** 2
                    error2_away_u += (away_goals - full_df['goals'].iloc[12474 + i]) ** 2
                    if full_df['goals'].iloc[i] > 0:
                        deviance_home_u += full_df['goals'].iloc[i] * math.log(full_df['goals'].iloc[i] / home_goals_predict) - (
                                full_df['goals'].iloc[i] - home_goals_predict)
                    else:
                        deviance_home_u += - (full_df['goals'].iloc[i] - home_goals_predict)
                    if full_df['goals'].iloc[12474 + i] > 0:
                        deviance_away_u += full_df['goals'].iloc[12474 + i] * math.log(
                            full_df['goals'].iloc[12474 + i] / away_goals_predict) - (
                                                   full_df['goals'].iloc[12474 + i] - away_goals_predict)
                    else:
                        deviance_away_u += - (full_df['goals'].iloc[12474 + i] - away_goals_predict)
                    matches_upper += 1
                    if prediction == full_df['result'].iloc[i]:
                        poisson_result_upper += 1
                    if home_goals == full_df['goals'].iloc[i] and away_goals == full_df['goals'].iloc[12474 + i]:
                        poisson_score_upper += 1
                result[prediction] += 1
                actual_result[full_df['result'].iloc[i]] += 1
                if home_goals > away_goals:
                    score_result['H'] += 1
                elif away_goals > home_goals:
                    score_result['A'] += 1
                else:
                    score_result['D'] += 1
            else:
                prediction, home_goals, away_goals, winner, home_goals_predict, away_goals_predict = prediction_poisson(
                    i, 'Unknown')
                if winner > accuracy_bound:
                    match_accuracy += 1
                    if full_df['goals'].iloc[i] > 0:
                        deviance_home_acc += full_df['goals'].iloc[i] * math.log(full_df['goals'].iloc[i] / home_goals_predict) - (
                                full_df['goals'].iloc[i] - home_goals_predict)
                    else:
                        deviance_home_acc += - (full_df['goals'].iloc[i] - home_goals_predict)
                    if full_df['goals'].iloc[12474 + i] > 0:
                        deviance_away_acc += full_df['goals'].iloc[12474 + i] * math.log(
                            full_df['goals'].iloc[12474 + i] / away_goals_predict) - (
                                                   full_df['goals'].iloc[12474 + i] - away_goals_predict)
                    else:
                        deviance_away_acc += - (full_df['goals'].iloc[12474 + i] - away_goals_predict)
                    if prediction == full_df['result'].iloc[i]:
                        result_accuracy += 1
                    if home_goals == full_df['goals'].iloc[i] and away_goals == full_df['goals'].iloc[12474 + i]:
                        score_accuracy += 1
                if ((full_df.team.iloc[rows] == full_df['team'].iloc[i]).sum() + (full_df.team.iloc[rows] == full_df['opponent'].iloc[i]).sum() <= lower_bound_sum
                    or (full_df.team.iloc[rows] == full_df['team'].iloc[i]).sum() <= lower_bound_individual
                    or (full_df.team.iloc[rows] == full_df['opponent'].iloc[i]).sum() <= lower_bound_individual):
                    matches_lower += 1
                    if prediction == full_df['result'].iloc[i]:
                        poisson_result_lower += 1
                    if home_goals == full_df['goals'].iloc[i] and away_goals == full_df['goals'].iloc[12474 + i]:
                        poisson_score_lower += 1
                    error1_home_l += (home_goals_predict - full_df['goals'].iloc[i]) ** 2
                    error1_away_l += (home_goals - full_df['goals'].iloc[i]) ** 2
                    error2_home_l += (away_goals_predict - full_df['goals'].iloc[12474 + i]) ** 2
                    error2_away_l += (away_goals - full_df['goals'].iloc[12474 + i]) ** 2
                    if full_df['goals'].iloc[i] > 0:
                        deviance_home_l += full_df['goals'].iloc[i] * math.log(full_df['goals'].iloc[i] / home_goals_predict) - (
                                full_df['goals'].iloc[i] - home_goals_predict)
                    else:
                        deviance_home_l += - (full_df['goals'].iloc[i] - home_goals_predict)
                    if full_df['goals'].iloc[12474 + i] > 0:
                        deviance_away_l += full_df['goals'].iloc[12474 + i] * math.log(
                            full_df['goals'].iloc[12474 + i] / away_goals_predict) - (
                                                   full_df['goals'].iloc[12474 + i] - away_goals_predict)
                    else:
                        deviance_away_l += - (full_df['goals'].iloc[12474 + i] - away_goals_predict)
                else:
                    error1_home_u += (home_goals_predict - full_df['goals'].iloc[i]) ** 2
                    error1_away_u += (home_goals - full_df['goals'].iloc[i]) ** 2
                    error2_home_u += (away_goals_predict - full_df['goals'].iloc[12474 + i]) ** 2
                    error2_away_u += (away_goals - full_df['goals'].iloc[12474 + i]) ** 2
                    if full_df['goals'].iloc[i] > 0:
                        deviance_home_u += full_df['goals'].iloc[i] * math.log(full_df['goals'].iloc[i] / home_goals_predict) - (
                                full_df['goals'].iloc[i] - home_goals_predict)
                    else:
                        deviance_home_u += - (full_df['goals'].iloc[i] - home_goals_predict)
                    if full_df['goals'].iloc[12474 + i] > 0:
                        deviance_away_u += full_df['goals'].iloc[12474 + i] * math.log(
                            full_df['goals'].iloc[12474 + i] / away_goals_predict) - (
                                                   full_df['goals'].iloc[12474 + i] - away_goals_predict)
                    else:
                        deviance_away_u += - (full_df['goals'].iloc[12474 + i] - away_goals_predict)
                    matches_upper += 1
                    if prediction == full_df['result'].iloc[i]:
                        poisson_result_upper += 1
                    if home_goals == full_df['goals'].iloc[i] and away_goals == full_df['goals'].iloc[12474 + i]:
                        poisson_score_upper += 1
                result[prediction] += 1
                actual_result[full_df['result'].iloc[i]] += 1
                if home_goals > away_goals:
                    score_result['H'] += 1
                elif away_goals > home_goals:
                    score_result['A'] += 1
                else:
                    score_result['D'] += 1
        print(f"The Poisson Distribution got {((poisson_result_lower + poisson_result_upper)/150) * 100} % of the results correct")
        print(f"The Poisson Distribution got {((poisson_score_lower + poisson_score_upper)/150) * 100} % of the scores correct")
        if matches_lower > 0:
            print(f"The Poisson Distribution got {(poisson_result_lower / matches_lower) * 100} % correct in the lower bound\n"
                f"The Poisson Distribution got {(poisson_score_lower / matches_lower) * 100} % of the scores correct in the lower bound")
            print(f"Lower bound home team goal difference: {math.sqrt(error1_home_l / matches_lower)} and {math.sqrt(error2_home_l / matches_lower)},\n"
                  f"Lower bound away team goal difference: {math.sqrt(error1_away_l / matches_lower)} and {math.sqrt(error2_away_l / matches_lower)}")
            print(f"Lower bound home team deviance: {2 * deviance_home_l}\n"
                  f"Lower bound away team deviance: {2 * deviance_away_l}")
            print(matches_lower)
        if matches_upper > 0:
            print(f"The Poisson Distribution got {(poisson_result_upper / matches_upper) * 100} % correct in the upper bound\n"
                f"The Poisson Distribution got {(poisson_score_upper / matches_upper) * 100} % of the scores correct in the upper bound")
            print(f"Upper bound home team goal difference: {math.sqrt(error1_home_u / matches_upper)} and {math.sqrt(error2_home_u / matches_upper)},\n"
                f"Upper bound away team goal difference: {math.sqrt(error1_away_u / matches_upper)} and {math.sqrt(error2_away_u / matches_upper)}")
            print(f"Upper bound home team deviance: {2 * deviance_home_u}\n"
                  f"Upper bound away team deviance: {2 * deviance_away_u}")
            print(matches_upper)
        if match_accuracy > 0:
            print(f"Prediction accuracy for matches with {accuracy_bound} or greater probability of winning: {(result_accuracy/match_accuracy)*100} % of {match_accuracy} matches")
            print(f"{(score_accuracy/match_accuracy)*100} % of {match_accuracy} matches were correct")
            print(f"Home team deviance for accuracy bound: {accuracy_bound}: {deviance_home_acc}")
            print(f"Away team deviance for accuracy bound: {accuracy_bound}: {deviance_away_acc}")
        print(dict(result))
        print(dict(score_result))
        print(dict(actual_result))

if __name__ == '__main__':
    compare_prediction_poisson_once()