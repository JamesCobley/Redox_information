#Download the raw data file from https://oximouse.hms.harvard.edu/download.html and upload into into the content folder in colab then run the script below to inspect change the file name as needed

import pandas as pd

df = pd.read_csv("site_all (3).csv")
print(df.info())
df.head()
