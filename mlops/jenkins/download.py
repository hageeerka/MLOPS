import pandas as pd

def download_data():
    url = 'https://github.com/ybifoundation/Dataset/raw/main/Salary%20Data.csv'
    salary = pd.read_csv(url)
    salary.to_csv("salary.csv", index=False)
    
    return True

download_data()