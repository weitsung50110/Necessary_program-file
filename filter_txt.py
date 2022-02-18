# importing library
import pandas as pd
import cv2
import numpy as np
#from imutils import paths


# Then loading csv file
for line in range(1000):
    df = pd.read_csv("new_baby/"+str(line)+".txt", sep=" ",header=None)
    #df = pd.read_csv("old_baby/999.txt", sep=" ",header=None)

    a = list(df[0])
    b = list(df[1])
    c = list(df[2])
    d = list(df[3])
    e = list(df[4])
    #print(a[0])


    
    #print(len(a))
    for i in range(len(a)):
        if a[i]<=55:
            print(str(line))
            print("GET")
            
        if a[i]>=77 and a[i]<=79:
            print(str(line))
            if a[i]==77:
                print("person")
            elif a[i]==78:
                print("man")
            elif a[i]==79:
                print("woman")
                
            print("GET")

        #a[i] = a[i] + 9
        #print(a[i])
       
    '''
    print("*********")
    df2={"222":a,"333":b,"444":c,"555":d,"666":e}
    mid_term_marks_df = pd.DataFrame(df2)


    mid_term_marks_df.to_csv("new_baby/"+str(line)+".txt", sep=" ",header=None,index=None)
    '''
      
