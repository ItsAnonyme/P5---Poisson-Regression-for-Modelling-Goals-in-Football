import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

data = pd.read_csv("Premier_League_All_Seasons.csv")
Team = {"Man City": 1, "Arsenal": 2,
"Man United": 3,
"Newcastle": 4,
"Liverpool": 5,
"Brighton": 6,
"Aston Villa": 7,
"Tottenham": 8,
"Brentford": 9,
"Fulham": 10,
"Crystal Palace": 11,
"Chelsea": 12,
"Wolves": 13,
"West Ham": 14,
"Bournemouth": 15,
"Nott'm Forest": 16,
"Everton": 17,
"Burnley": 18,
"Sheffield United": 19,
"Luton": 20}


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
    for i in range(11564, 11944):
        if matchcheck(data.iloc[i, 1], data.iloc[i, 2], data.iloc[i, 3], data.iloc[i, 4]) == 1:
            p+=1
        else:
            q+=1

if __name__ == "__main__":
    s=0
    for i in range(11564, 11944):
        if data.iloc[i, 3] == data.iloc[i, 4]:
            s+=1
        else:
            s+=0

table = [['Succes','Succes%'],
         [p,p/380]]
print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))      
        
        
        
        

