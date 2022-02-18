# importing library
import pandas as pd
import cv2
import numpy as np

# Then loading csv file
df = pd.read_csv('public_training_data.csv')

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
## converting list into string and then joining it with space
#b = ' '.join(str(e) for e in a)

# printing result
#print(a)
sizeF = len(a)
print(sizeF)

index=0
while (index < sizeF):
   filename = "public_training_data/public_training_data" + a[index] + ".jpg"
   img = cv2.imread(filename)

   label = b[index] 

   tr_x = int(c[index])   
   tr_y = int(d[index])
   br_x = int(e[index])   
   br_y = int(f[index])
   bl_x = int(g[index])   
   bl_y = int(h[index])
   tl_x = int(i[index])   
   tl_y = int(j[index])
   print(filename, label, tr_x, tr_y, br_x, br_y, bl_x, bl_y, tl_x, tl_y)
   
   # Polygon corner points coordinates
   pts = np.array([[tr_x, tr_y], [br_x, br_y], 
                   [bl_x, bl_y], [tl_x, tl_y]],
                    np.int32)
   '''
   pts.reshape((-1, 1, 2))
   这会将形状从(4,2)更改为(4,1,2)，
   这与多个cv2函数使用的形状一致。
   例如，如果要使用findContours查找轮廓，
   则输出轮廓的形状为(x，1，y)。
   '''
   pts = pts.reshape((-1, 1, 2))   #I understand this makes pts.shape to be (4,1,2)
   isClosed = True
   # Blue color in BGR
   color = (255, 0, 0)
   # Line thickness of 2 px
   thickness = 2
   # Using cv2.polylines() method
   # Draw a Blue polygon with 
   # thickness of 2 px
   img = cv2.polylines(img, [pts], isClosed, color, thickness)

   # font 
   font = cv2.FONT_HERSHEY_SIMPLEX 
   # org 
   org = (tl_x, tl_y - 30) 

   # fontScale 
   fontScale = 2
   # Red color in BGR 
   color = (0, 0, 255) 
   # Line thickness of 2 px 
   thickness = 2
   # Using cv2.putText() method 
   img = cv2.putText(img, label, org, font,  
                   fontScale, color, thickness, cv2.LINE_AA)

   cv2.imshow(filename, img)
   if cv2.waitKey(0) & 0xFF == ord('q'):
      break

   index = index + 1
   cv2.destroyWindow(filename)

cv2.destroyAllWindows()

# converting 'label' column into list
#d = list(df['label'])
#sizeL = len(d)
#print(sizeL)

# another way for joining used
#e = '\n'.join(map(str, d))

# printing result
#print(d)

