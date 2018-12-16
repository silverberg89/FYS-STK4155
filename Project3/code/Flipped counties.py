import numpy as np
import pandas as pd

# Import CSV-file
data = pd.read_table('Data/US_Election.csv', sep = ';', header = 0, encoding = 'latin1')

# 2012 Election 
DEM_2012 = np.array(data.iloc[:, -2])
DEM_2012 = np.where(DEM_2012 > 0.5, 1, 0)

# 2016 Election
DEM_2016 = np.array(data.iloc[:, -4])
DEM_2016 = np.where(DEM_2016 > 0.5, 1, 0)

# Counties indices 
flipped_counties_indices = np.where(DEM_2012 != DEM_2016)[0]

# Binary classification of the counties
n = len(DEM_2012)
flipped_counties = np.zeros((n,))
flipped_counties[flipped_counties_indices] = 1
    
# Write to file 
#f = open("Data/flipped_counties.txt", "w+")
#for i in range(n):
#     f.write('%d \n' % flipped_counties[i])
#f.close() 