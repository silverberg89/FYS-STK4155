import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def import_election_data(Year):
    
    """
    Year: Election year. 2012 or 2016.
    Year 2012 is split in training and test data. 
    """

    # Import CSV-file
    data = pd.read_table('Data/US_Election.csv', sep = ';', header = 0, encoding = 'latin1')

    # Construct the design matrix with indexation 
    X = np.array(data.iloc[:, :-4]) # Dependant variables
    scale = StandardScaler()
    scale.fit(X)
    X = scale.transform(X)

    # 2012 Election 
    if Year == 2012:
       DEM_2012 = np.array(data.iloc[:, -2])
       DEM_2012 = np.where(DEM_2012 > 0.5, 1, 0)
       return train_test_split(X, DEM_2012)
    
    # 2016 Election
    if Year == 2016:
        DEM_2016 = np.array(data.iloc[:, -4])
        DEM_2016 = np.where(DEM_2016 > 0.5, 1, 0)
        return X, DEM_2016

