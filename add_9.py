# importing library
import pandas as pd
import cv2
import numpy as np
#from imutils import paths


# Then loading csv file
for line in range(1000):
    df = pd.read_csv("old_baby/"+str(line)+".txt", sep=" ",header=None)
    #df = pd.read_csv("old_baby/999.txt", sep=" ",header=None)

    a = list(df[0])
    b = list(df[1])
    c = list(df[2])
    d = list(df[3])
    e = list(df[4])
    #print(a[0])


    print(len(a))
    for i in range(len(a)):
        a[i] = a[i] + 9
        #print(a[i])
       
    print("*********")
    df2={"222":a,"333":b,"444":c,"555":d,"666":e}
    mid_term_marks_df = pd.DataFrame(df2)


    mid_term_marks_df.to_csv("new_baby/"+str(line)+".txt", sep=" ",header=None,index=None)
          
      
'''
# converting ;filename' column into list
a = list(df['filename'])
b = list(df['label'])
c = list(df['top right x'])
d = list(df['top right y'])
e = list(df['bottom right x'])
f = list(df['bottom right y'])
g = list(df['bottom left x'])
h = list(df['bottom left y'])
i = list(df['top left x'])
j = list(df['top left y'])

#print(a[11976])
#print(b[11976])
for line in a:
   if line[0:2] == "'=":
      print(line)

steelNames = open('darknet/steel/steel.names')
names = []
for line in steelNames:
    names.append(line.split("\n")[0])
#print(names)
steelNames.close()

imagePaths = list(paths.list_images("steel/labels/"))
#print(imagePaths)
imPaths = sorted(imagePaths)
#print(imPaths)
data_size = len(imPaths)
print(data_size)

with open('differentLabel.txt', 'w') as dlf:
   ind=0
   while (ind < data_size):
      filename = imPaths[ind].split(".")[0].split("/")[-1]

      labelTxt = "steel/labels/" + filename + ".txt"
#      print(labelTxt)
   
      labelFileNames = open(labelTxt)
      labelText = []
      for line in labelFileNames:
         element = line.split(" ")
         tag = int(element[0])
         bx = float(element[1])
         by = float(element[2])
         bw = float(element[3])
         bh = float(element[4])
         #print(tag, bx, by, bw, bh)
         labelText.append([names[tag], bx])
      #print(labelText)
      labelFileNames.close()

      #labelText_sort = sorted(labelText, key = lambda labelText : labelText[1])
      #print(labelText)

      labelString = ""
      for (item, x) in labelText:
         labelString += item
      print (labelString)
      if filename[0] == "=":
         print(filename)
         filename = "'" + filename + "'"
         print(filename)
      res = a.index(filename)
      print(res)
      label = b[res]
#      print(a[res], label, c[res], d[res], e[res], f[res], g[res], h[res], i[res], j[res])

      len1=len(labelString)
      len2 = len(label)
      print(len1, len2)
      #if labelString != label:
      if len1!= len2:
         #reverseLabelString = labelString[::-1]
         #print("reverse Label: " + reverseLabelString)
         #if reverseLabelString != label:
         dlf.write(labelTxt)
         dlf.write(' ')
         dlf.write(label)
         dlf.write(' ')
         dlf.write(labelString)
         dlf.write('\n')
      
      ind += 1

dlf.close()

'''