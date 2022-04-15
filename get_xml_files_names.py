import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join


#file = open('image/2022train.txt', 'w')


#把所有的xml檔案的名字存到一個資料夾裡面
#此例是存在new/name_classes.txt裡面


path = "train/216/" #資料夾目錄
files= os.listdir(path) #得到資料夾下的所有檔名稱


for file in files: #遍歷資料夾
	if "xml" in file:   #xml is keyword
		print(file.replace(".xml",""))
		
		with open("new/name_classes.txt", 'a') as f:
				f.write(file.replace(".xml","")+"\n") #只保留名字 不要.xml的檔名

		