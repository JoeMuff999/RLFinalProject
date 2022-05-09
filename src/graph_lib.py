import matplotlib.pyplot as plt
import numpy as np

def rewards_over_time():
    file = open('../data/total_rewards-150.txt')
    data = []
    for idx, line in enumerate(file):
        line = line.replace('[', '')
        line = line.replace(']', '')
        data.append([])
        for elem in line.split(','):
            data[idx].append(round(float(elem), 2))
    xaxis = [i for i in range(len(data[0]))]
    for d in data:
        plt.plot(xaxis, d)
    plt.show()

def rewards_over_time_confidence():
    file = open('../data/total_rewards-150.txt')
    data = []
    for idx, line in enumerate(file):
        line = line.replace('[', '')
        line = line.replace(']', '')
        data.append([])
        for elem in line.split(','):
            data[idx].append(round(float(elem), 2))
    xaxis = [i for i in range(len(data[0]))]

    

    smart_data = [0.0 for _ in range(len(data[0]))]

    plt.show()

rewards_over_time()