
Visualisation of Covid database, available at https://www.kaggle.com/gpreda/covid-world-vaccination-progress


```python

import pandas as pd
from pandas_profiling import ProfileReport
from IPython.display import HTML

df = pd.read_csv("country_vaccinations.csv")

profile = ProfileReport(df)

#If you want an html report 
#profile.to_file(output_file='rapport.html')

HTML(filename="rapport.html")

```
