## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
Developed by : Priyanka K
Reg No : 212223230162
```

```python
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/861ca668-54a2-486a-a75b-4db3e46d361c)

```python
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/c46bcfc8-24cb-418d-b65f-e5c7c189bf9e)



```python
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![Screenshot 2024-10-01 112751](https://github.com/user-attachments/assets/13c12a60-747c-4cb8-a926-69e7cc1cfccf)



```python
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/eaa14681-695e-42af-b895-eb82314d26b1)



```python
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```
![image](https://github.com/user-attachments/assets/0b893acb-e20b-46e5-b032-6fd3e7b58145)


```python
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/25c8e2ff-3f5c-44b9-a873-b9a550abdd32)

```python
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/d44d08aa-be43-488d-984d-391f4f0042d8)



```python
pip install --upgrade category_encoders
```
![Screenshot 2024-10-01 113423](https://github.com/user-attachments/assets/00fa038c-43ae-45b0-9f04-2367a472c37c)



```python
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![image](https://github.com/user-attachments/assets/d06d39c7-08ad-401d-87a6-d75adc952c76)



```python
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/573e1f48-66ff-4c93-bcd9-c1a079422901)


```python
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/d551e20d-ddf0-4237-aef0-8e51895b535e)



```python
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/cfc2654f-eb7d-428e-896a-9ee6eccf26fc)



```python
df.skew()
```
![Screenshot 2024-10-01 113655](https://github.com/user-attachments/assets/45d05009-d673-4d44-a8f0-bb839727254e)



```python
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/e54c07d4-0f38-4530-a630-291190b49078)



```python
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/eae57994-bf82-4abd-9a3b-fae00cc2eff4)


```python
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/61b4bd41-09f9-4f04-8dc8-5b45405d4139)


```python
np.square(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/a3a1d80d-1499-4a91-b9a5-bb043b911c5a)


```python
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/7f960eba-b8e5-43d3-a37b-24f30d5a4f97)


```python
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df
```
![image](https://github.com/user-attachments/assets/6edb8d31-a26f-436f-ad7c-baec90af188a)


```python
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/f3710ce9-e1aa-4770-a953-7d0ffc7a42b2)



```python
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/3fdfa646-b1b5-44e6-95f8-3cc6241c9198)



```python
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```



```python
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

![Screenshot 2024-10-01 114501](https://github.com/user-attachments/assets/a7fd36c4-f9b5-439c-ac27-e669ba86ed9a)

```python
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/8feebab1-2479-4896-9999-f0832d1b0a7b)




## RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
       
