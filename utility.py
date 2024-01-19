import os
import csv

def read_as_csv(csv_paths):
    x = []
    y = []
    
    with open(csv_paths, 'r') as file:
        csvreader = csv.reader(file)
        next(csvreader)
        for i in csvreader:
            x.append(i[1])
            y.append(i[2])
        return (x,y)
    

