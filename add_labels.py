# importing library
import csv
import pandas as pd

# Then loading csv file
df = pd.read_csv('submission_template.csv')
	
a = list(df['id'])  #圖片名稱
b = list(df['text']) #圖片label text

sizeF = len(a)
print(sizeF)

index=0
while (index < sizeF):

	if a[index] == "圖片名稱":
		
		b[index] = "label text" #把label答案丟進矩陣裡面
	
	index = index + 1	
			
dataframe = pd.DataFrame({"id":a,"text":b})
dataframe.to_csv("test.csv",index=False) #存成新的一個.csv檔案

