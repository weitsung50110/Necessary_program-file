import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join



#把00001 ~ 00009的檔案轉成.txt
'''

for i in range(10):
    for j in range(200000):
        result = os.path.isfile("0000"+str(i+1)+"_"+str(j)+".jpg")
        #print(result)
        if(result==True):
            with open("000/0000"+str(i+1)+"_"+str(j)+".txt", 'w') as f:
                f.write(str(i)+" 0.501953 0.500885 0.996094 0.998230")
                print(str(i)+" 0.501953 0.500885 0.996094 0.998230" + "000/0000"+str(i+1)+"_"+str(j)+".txt")
'''

#把00010 ~ 00099的檔案轉成.txt

with open("labels.txt", 'r') as f:
    with open("Ai_city_classes.txt", 'a') as a:
        a.write("names: [")
        for line in f.readlines():
            print(line.strip())
            a.write("\'"+line.strip()+"\',")
        a.write("] ")
