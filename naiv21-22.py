import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

data = pd.read_csv("Premier_League_All_Seasons.csv")
Team = {"Man City": 1, "Man United": 2,
"Liverpool": 3,
"Chelsea": 4,
"Leicester": 5,
"West Ham": 6,
"Tottenham": 7,
"Arsenal": 8,
"Leeds": 9,
"Everton": 10,
"Aston Villa": 11,
"Newcastle": 12,
"Wolves": 13,
"Crystal Palace": 14,
"Southampton": 15,
"Brighton": 16,
"Burnley": 17,
"Norwich": 18,
"Watford": 19,
"Brentford": 20}


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
    for i in range(10804, 11184):
        if matchcheck(data.iloc[i, 1], data.iloc[i, 2], data.iloc[i, 3], data.iloc[i, 4]) == 1:
            p+=1
        else:
            q+=1

if __name__ == "__main__":
    s=0
    for i in range(10804, 11184):
        if data.iloc[i, 3] == data.iloc[i, 4]:
            s+=1
        else:
            s+=0

table = [['Succes','Succes%'],
         [p,p/380]]
print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))           
        
        
        
        

