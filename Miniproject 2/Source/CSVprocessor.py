'''
Plotting csv files.

Author: Nicoline Louise Thomsen
'''

import csv as csv_libary
import matplotlib.pyplot as plt


class CSVprocessor():
    
    def __init__(self, filename, n_columns=2):
        self.filename = filename
        self.n_columns = n_columns

        self.data = self.get_data()


    def get_data(self):

        data = [[] for i in range(self.n_columns)]
        
        with open(self.filename,'r') as csvfileQuick:
            file = csv_libary.reader(csvfileQuick, delimiter=',')
            next(file)  # Skip first line / header
            for row in file:
                for i in range(self.n_columns):
                    data[i].append(row[i])

        return data

    
    def extract_float_columns(self, colum_ids):

        data = [[] for i in range(len(colum_ids))]

        for i, ID in enumerate(colum_ids):
            for j in range(len(self.data[ID])):
                data[i].append(float(self.data[ID][j]))

        return data
    
    
    def plot_float_data(self, title='Title', xlabel='x', ylabel='y', column_name='none'):
        
        dataset = self.extract_float_columns(range(len(self.data)))
        t = dataset[0]
        data = dataset[1:]
        
        for i, column in enumerate(data):
            plt.plot(t, column, label = column_name + str(i), markersize=1)
    
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        if (column_name != 'none'):
            plt.legend()
        
        plt.show()



if __name__ == '__main__':

    CSV = CSVprocessor('../Dataset/DJIFlightRecord_2021-03-18_[13-04-51]-TxtLogToCsv.csv', 274)
    gps = CSV.extract_float_columns([12, 13, 15])

    print(gps[2])

