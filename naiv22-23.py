import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

data = pd.read_csv("Premier_League_All_Seasons.csv")
Team = {"Man City": 1, "Liverpool": 2,
"Chelsea": 3,
"Tottenham": 4,
"Arsenal": 5,
"Man United": 6,
"West Ham": 7,
"Leicester": 8,
"Brighton": 9,
"Wolves": 10,
"Newcastle": 11,
"Crystal Palace": 12,
"Brentford": 13,
"Aston Villa": 14,
"Southampton": 15,
"Everton": 16,
"Leeds": 17,
"Fulham": 18,
"Bournemouth": 19,
"Nott'm Forest": 20}


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
    for i in range(11184, 11564):
        if matchcheck(data.iloc[i, 1], data.iloc[i, 2], data.iloc[i, 3], data.iloc[i, 4]) == 1:
            p+=1
        else:
            q+=1

table = [['Succes','Succes%'],
         [p,p/380]]
print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))        
        
        
        

