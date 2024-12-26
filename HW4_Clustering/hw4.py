import scipy
import numpy as np
import matplotlib.pyplot as plt
import math
import csv

def load_data(filepath):
    pokedex = [];
    with open(filename) as file:
        pokedex = csv.DictReader(file)
        
        
def calc_features(row):
    values = ['Attack', 'Sp. Atk', 'Speed', 'Defense', 'Sp. Def', "HP"]
    temp = []
    for i in values:
        temp.append(int(row.get(i)))
    return np.array(temp)
       
def hac(features):
    # yeah, no
    pass

def imshow_hac(Z):
    plt.show()      