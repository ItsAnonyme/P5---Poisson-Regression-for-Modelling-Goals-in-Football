import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Choose your teams
HomeTeam = "Man United"
AwayTeam = "Liverpool"

# Where we get our data
Data = pd.read_csv("Premier_League_All_Seasons.csv",skipfooter=430,engine='python')
List = pd.read_csv("Premier_League_All_Seasons.csv",skiprows=range(1,11944),skipfooter=50,engine='python')

def calculate_weights(dates, x):
    dates = pd.to_datetime(dates, errors='coerce')
    latest_date = dates.max()
    diff = ((latest_date - dates).dt.days)
    weights = np.exp(-x * diff)
    return weights

def generalized_linear_model_binomial():
    home_df = pd.DataFrame(
        data={"team": Data.HomeTeam, "opponent": Data.AwayTeam, "is_draw": (Data.FTHG == Data.FTAG).astype(int), "home": 1,
              "date": Data.Date})
    away_df = pd.DataFrame(
        data={"team": Data.AwayTeam, "opponent": Data.HomeTeam, "is_draw": (Data.FTAG == Data.FTHG).astype(int), "home": 0,
              "date": Data.Date})

    df = pd.concat([home_df, away_df]).reset_index(drop=True)
    # Applying Poisson Distribution
    df["weights"] = calculate_weights(df["date"], 0.004)
    model_binomial = smf.glm(data=df, family=sm.families.Binomial(),
                            formula="is_draw ~ home + team + opponent", freq_weights=df["weights"]).fit()

    return model_binomial

def prediction_draw(model, HomeTeam, AwayTeam):
    df = pd.DataFrame({"team": [HomeTeam], "opponent": [AwayTeam], "home": 1} )

    #the probability of draw
    p = model.predict(df)[0]
    return(p)

def model_winning():
    no_draw_df = Data[Data.FTHG != Data.FTAG].copy()

    home_df = pd.DataFrame(
        data={"team": no_draw_df.HomeTeam, "opponent": no_draw_df.AwayTeam, "home_win": (no_draw_df.FTHG > no_draw_df.FTAG).astype(int), "home": 1, "date": no_draw_df.Date})
    away_df = pd.DataFrame(
        data={"team": no_draw_df.AwayTeam, "opponent": no_draw_df.HomeTeam, "home_win": (no_draw_df.FTAG > no_draw_df.FTHG).astype(int), "home": 0, "date": no_draw_df.Date})

    df = pd.concat([home_df, away_df]).reset_index(drop=True)

    df["weights"] = calculate_weights(df["date"], 0.004)
    model_binomial = smf.glm(data=df, family=sm.families.Binomial(),
                            formula="home_win ~ home + team + opponent", freq_weights=df["weights"]).fit()
    return model_binomial

def prediction_home_win(model, HomeTeam, AwayTeam):
    df = pd.DataFrame({"team": [HomeTeam], "opponent": [AwayTeam], "home": 1} )

    #the probability of a home team win
    p = model.predict(df)[0]
    return(p)

def prediction_result(model1, model2, HomeTeam, AwayTeam, t):
    p_draw = prediction_draw(model1, HomeTeam, AwayTeam)
    if p_draw > t:
        return 'D'
    else:
        p_home_win = prediction_home_win(model2, HomeTeam, AwayTeam)
        if p_home_win > 1 - p_home_win:
            return 'H'
        else:
            return 'A'

def glm_test(t):
    prediction, draw_wrong, draw_correct, draw_total = 0, 0, 0, 0
    home, away = 0, 0
    model1 = generalized_linear_model_binomial()
    model2 = model_winning()
    for i in range(len(List)):
        result = prediction_result(model1, model2, List['HomeTeam'].iloc[i], List['AwayTeam'].iloc[i], t)
        if result == 'H':
            home += 1
        if result == 'A':
            away += 1
        if List['FTR'].iloc[i] == 'D':
            draw_total += 1
        if result == List['FTR'].iloc[i]:
            prediction += 1
        if result == "D" and result == List['FTR'].iloc[i]:
            draw_correct += 1
        elif result == "D" and result != List['FTR'].iloc[i]:
            draw_wrong += 1
    print(f"Results estimated correctly: {(prediction / len(List)) * 100}")
    print(f"Distribution is Home: {home}, Away: {away}, Draw: {draw_correct + draw_wrong}")
    print(
        f"Draw estimated correctly: {draw_correct}, draw estimated wrong: {draw_wrong}, procent draw estimated correctly: {draw_correct / draw_total * 100}")
    return ((prediction/len(List))*100), draw_correct, draw_total


def optimal_t(n):
    nums = np.linspace(0.2, 0.5, n)
    results = []
    draw_correct = []
    draw_total = []

    for t in nums:
        test = glm_test(t)
        results.append(test[0])
        draw_correct.append(test[1])
        draw_total.append(test[2])

    best_idx = np.argmax(results)
    return nums[best_idx], results[best_idx], draw_correct[best_idx], draw_total[best_idx], draw_correct[best_idx]/draw_total[best_idx]*100, results

optimal = optimal_t(100)
nums = np.linspace(0.2, 0.5, 100)

plt.figure(figsize=(10,6))
plt.plot(nums, optimal[5], marker='o', label="Match Result Accuracy")
plt.title("Accuracy vs t")
plt.xlabel("t")
plt.ylabel("Prediction Accuracy (%)")
plt.grid(True)

    # Highlight the best decay rate
plt.scatter(optimal[0], optimal[1], s=120, marker='o', edgecolors='black',
            label=f"Best decay rate = {optimal[0]:.4f}\nAccuracy = {optimal[1]:.4f}%")

plt.legend()
plt.tight_layout()
plt.show()
print(nums)


if __name__ == "__main__":
    #glm_test(0.2788)
    print(optimal_t(20))
