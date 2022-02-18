# importing library
import csv
import pandas as pd

# Then loading csv file
df = pd.read_csv('submission_template.csv')
df2 = pd.read_csv('submission.csv')
	
a = list(df['id'])  #圖片名稱
b = list(df['text']) #圖片label text

a2 = list(df2['id'])  #圖片名稱
b2 = list(df2['text']) #圖片label text

sizeF = len(a)
print(sizeF)

index=0
j=0
while (index < sizeF):
	for j in range(6000):
		
		if ( a[index] == a2[j] ):
			b[index] = b2[j] #把label答案丟進矩陣裡面
			
		j = j + 1
	
	index = index + 1	
			
dataframe = pd.DataFrame({"id":a,"text":b})
dataframe.to_csv("test555.csv",index=False) #存成新的一個.csv檔案

