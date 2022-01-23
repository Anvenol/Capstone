import pandas as pd
import numpy as np


df = pd.DataFrame(data=np.arange(20).reshape(5,4), columns=['a', 'b', 'c', 'd'])
df.rename(columns={'a':"???"},inplace=True)
print(df)