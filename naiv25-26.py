import pandas as pd
from tabulate import tabulate

data = pd.read_csv("C:/Users/75tho/P5/PythonKode/Premier League All Seasons.csv")
Team = {"Liverpool": 1, "Arsenal": 2,
"Man City": 3,
"Chelsea": 4,
"Newcastle": 5,
"Aston Villa": 6,
"Nott'm Forest": 7,
"Brighton": 8,
"Bournemouth": 9,
"Brentford": 10,
"Fulham": 11,
"Crystal Palace": 12,
"Everton": 13,
"West Ham": 14,
"Man United": 15,
"Wolves": 16,
"Tottenham": 17,
"Leeds": 18,
"Burnley": 19,
"Sunderland": 20}


def matchcheck(H,A,FTHG,FTAG):
    p=0
    q=0
    x=0
    if FTHG > FTAG:
        if Team[H]-Team[A]<-x:
            p +=1
        else:
            q +=1
    elif FTHG == FTAG:
        if abs(Team[H]-Team[A]) <=x:
            p+=1
        else:
            q+=1
    else:
        if Team[A]-Team[H] <-x :
            p +=1
        else:
            q +=1
    return p

if __name__ == "__main__":
    p=0
    q=0
    for i in range(12325, 12474):
        if matchcheck(data.iloc[i, 1], data.iloc[i, 2], data.iloc[i, 3], data.iloc[i, 4]) == 1:
            p+=1
        else:
            q+=1

table = [['Succes','Succes%'],
         [p,p/150]]
print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))      
        
        
        
        

