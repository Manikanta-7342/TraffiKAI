from Audio_Emergency.model_audio import *
from Opencv_Emergency.model_image import *
import time,sys

tick=time.time()
east_path = sys.argv[1]
south_path = sys.argv[2]
west_path = sys.argv[3]
north_path = sys.argv[4]

li1=emergency_audio(east_path,south_path,west_path,north_path,'Audio_Emergency/')
li2=emergency_image(east_path,south_path,west_path,north_path,'Opencv_Emergency/')
li=[]

for i in range(0,3):
    i=int(i)
    t=li1[i]*0.8+li2[i]*0.2
    li.append(1) if t>0.5 else li.append(0)


print(li)
with open('emer.txt', 'w') as file:
    file.write(str(li))

print('\n\n', 'Time taken: ', time.time() - tick)
print(li1)
print(li2)