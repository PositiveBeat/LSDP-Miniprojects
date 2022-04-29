'''
Plotting csv files for case d).

Author: Nicoline Louise Thomsen
'''

import csv
import matplotlib.pyplot as plt


def plotCSV(filename, column_name, columns=2):
    
    t = []
    data = [[] for i in range(columns - 1)]
    
    with open(filename,'r') as csvfileQuick:
        plots = csv.reader(csvfileQuick, delimiter=',')
        next(plots)
        for row in plots:
            t.append(int(row[0]))
            
            for i in range(columns - 1):
                data[i].append(float(row[i+1]))
    
    
    for i, column in enumerate(data):
        plt.plot(t, column, 'rx', label = column_name + str(i), markersize=1)
    
    plt.xlabel('Time steps')
    plt.ylabel('Sync')
    plt.title('Sync')
    # plt.ylabel('Average heading [rad]')
    # plt.title('Average heading over time')
    # plt.legend()
    
    plt.show()
    


if __name__ == '__main__':
    plotCSV('logs/data_sync40.csv', 'drone', columns=101)
