# EXNO-3-DS

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
DEeveloped by: Priyanka K
Register No: 212223230162
```
```
import pandas as pd
df = pd.read_csv("Encoding_data.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/3c4708b7-1e8e-404f-bfa4-ad9828da4d8b)

```
df.tail()
```
![exp3](https://github.com/user-attachments/assets/69993b8c-66db-4689-8250-a37230bbcbae)

```
df.describe()
```
![image](https://github.com/user-attachments/assets/b470ef07-1cd5-4484-b711-9f61520d7e56)

```
df.info()
```
![image](https://github.com/user-attachments/assets/b0a38927-ebd6-4b90-abd7-5b21436f8901)

```
df.shape
```
![3](https://github.com/user-attachments/assets/576d78f3-6747-422a-9a3f-31b79b218db7)

```
df
```
![image](https://github.com/user-attachments/assets/f62c2a51-ccd7-4888-bd01-1e29443f765f)

```
#ordinal encoder
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
pm=['Hot', 'Warm','Cold']
oe=OrdinalEncoder(categories=[pm])
oe.fit_transform(df[["ord_2"]])
```
![arr](https://github.com/user-attachments/assets/af3fcacf-b7eb-457e-98d9-3889b41d7e2a)

```
df['bo2']=oe.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/f4824aa3-72e2-4354-b39a-14d7bc6c5da8)

```
#label Encoder
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/43d81ec8-49ab-43ef-888b-d7f097a39452)

```
#One hot encoder
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse = False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/b1ddad21-90b8-4159-9548-8eca78a7dae5)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/a0ea2434-a534-4afa-8b91-b17617c06e5a)

```
pip install --upgrade category_encoders
from category_encoders import BinaryEncoder
df= pd.read_csv("data.csv")
df
```
![image](https://github.com/user-attachments/assets/421dc1e1-fcb5-4202-87da-4517a20956d6)
```
#binary encoder
be = BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb

```
![image](https://github.com/user-attachments/assets/9f44b784-9b99-41d2-8f29-39d3693cbbae)

```
#target encoder
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new = te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/user-attachments/assets/f91f18ce-7e69-4cee-81a3-6bcf54b48958)

```
#Feature Transformation
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/3b1ec929-ad0e-47fb-b39a-03d9c228d6e1)

```
df.info()
```
![image](https://github.com/user-attachments/assets/327ed0d1-2f8f-41c5-9da9-125c79d5d1ff)

```
df.describe()
```
![image](https://github.com/user-attachments/assets/5f5c8f7a-49be-48dc-86c1-400a0214618d)

```
df.size
```
![image](https://github.com/user-attachments/assets/4af8996e-d807-4ba2-8def-1ddcddca8d4d)

```
df.skew()
```
![image](https://github.com/user-attachments/assets/8be208a6-db9f-4d23-abb3-7ccd8661225a)

```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/78aafe8a-29e8-4602-94cf-04d5a6412c76)

```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/11e16ca2-36cc-4bb4-9ae2-8602167b6df5)

```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/2b3b9469-a49c-4285-8c53-6158025da034)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/b2ee6d10-0ca8-4c43-8a5c-d1e425318246)
```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/8715ef25-6571-4a90-b9a1-5d6ad36ccaf1)

```
df["Moderate Negative Skew_yeojohnson"],parameters =stats.yeojohnson(df["Moderate Negative Skew"])
df
```
![image](https://github.com/user-attachments/assets/fda58aed-306d-4447-b66a-db7cdd0b233a)

```
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/e4826368-b926-4d18-bc8f-7ff61372cc92)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])


sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/189e9be3-76c4-41ca-a7cf-63764cebec7b)

```
df
```
![image](https://github.com/user-attachments/assets/9a74678c-3e7c-4d20-96cf-c95e1b37707a)

```
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/cdccbea2-1a33-4f3b-9ba3-95a12ac42751)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/c2a44890-9aee-4a59-808c-e5e43bd57817)

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/b42704ee-86d4-4dee-8d41-544235bf59e3)


```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/13242ffc-05a2-467d-ba27-0c3555419b26)


# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
