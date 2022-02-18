# importing library
import pandas as pd
import cv2
import numpy as np
#from imutils import paths


a56 = 0
a57 = 0
a58 = 0
a59 = 0
a60 = 0
a61 = 0
a62 = 0
a63 = 0
a64 = 0
a65 = 0
a66 = 0
a67 = 0
a68 = 0
a69 = 0
a70 = 0
a71 = 0
a72 = 0
a73 = 0
a74 = 0
a75 = 0
a76 = 0
a80 = 0
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

        if a[i]==56:
            a56 = a56 + 1
        elif a[i]==57:
            a57 = a57 + 1
        elif a[i]==58:
            a58 = a58 + 1
        elif a[i]==59:
            a59 = a59 + 1
        elif a[i]==60:
            a60 = a60 + 1
        elif a[i]==61:
            a61 = a61 + 1
        elif a[i]==62:
            a62 = a62 + 1
        elif a[i]==63:
            a63 = a63 + 1
        elif a[i]==64:
            a64 = a64 + 1
        elif a[i]==65:
            a65 = a65 + 1
        elif a[i]==66:
            a66 = a66 + 1
        elif a[i]==67:
            a67 = a67 + 1
        elif a[i]==68:
            a68 = a68 + 1
        elif a[i]==69:
            a69 = a69 + 1
        elif a[i]==70:
            a70 = a70 + 1
        elif a[i]==71:
            a71 = a71 + 1
        elif a[i]==72:
            a72 = a72 + 1
        elif a[i]==73:
            a73 = a73 + 1
        elif a[i]==74:
            a74 = a74 + 1
        elif a[i]==75:
            a75 = a75 + 1
        elif a[i]==76:
            a76 = a76 + 1
        elif a[i]==80:
            a80 = a80 + 1

        #a[i] = a[i] + 9
        #print(a[i])

#print('c_r','c_g','c_b','c_y','c_k','c_w','c_o','c_s','c_z','b_d','b_n','x_d','x_n','s_d','s_n','t_d','t_n','m_d','m_n','z_d','z_n','c_a')

print('c_r', a56 , "個")
print('c_g', a57 , "個")
print('c_b', a58 , "個")
print('c_y', a59 , "個")
print('c_k', a60 , "個")
print('c_w', a61 , "個")
print('c_o', a62 , "個")
print('c_s', a63 , "個")
print('c_z', a64 , "個")
print('b_d', a65 , "個")
print('b_n', a66 , "個")
print('x_d', a67 , "個")
print('x_n', a68 , "個")
print('s_d', a69 , "個")
print('s_n', a70 , "個")
print('t_d', a71 , "個")
print('t_n', a72 , "個")
print('m_d', a73 , "個")
print('m_n', a74 , "個")
print('z_d', a75 , "個")
print('z_n', a76 , "個")
print('c_a', a80 , "個")
