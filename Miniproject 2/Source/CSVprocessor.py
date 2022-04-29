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
            next(file)
            for row in file:
                for i in range(self.n_columns):
                    data[i].append(row[i])

        return data

    
    def extract_columns(self, colum_ids):

        data = [[] for i in range(len(colum_ids))]

        for i, ID in enumerate(colum_ids):
            for j in range(len(self.data[ID])):
                data[i].append(float(self.data[ID][j]))

        return data

    


if __name__ == '__main__':

    CSV = CSVprocessor('../Dataset/DJIFlightRecord_2021-03-18_[13-04-51]-TxtLogToCsv.csv', 274)
    gps = CSV.extract_columns([12, 13, 15])

    print(gps[2])

