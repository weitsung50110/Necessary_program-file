import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import shutil


#file = open('image/2022train.txt', 'w')


#把所有的的檔案"複製"到新的together資料夾裡面 但不要classes的檔案
#此例是存在together裡面


for line in range(219):

	path = "train/"+str(line)+"/" #資料夾目錄
	files= os.listdir(path) #得到資料夾下的所有檔名稱


	for file in files: #遍歷資料夾
		if "classes" in file:   #jpg is keyword
			print("break")
		else:
			#print(file.replace(".xml",""))
			print(file)
			
			
			src=r'train/'+str(line)+'/'+file
			des=r'together'
			shutil.copy2(src, des)
			
