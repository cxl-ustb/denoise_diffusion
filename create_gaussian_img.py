import cv2
import os
import numpy as np

img_dir=os.listdir('/data/dataset/celeba_hq_256')
img_dir=sorted(img_dir)

res_dir='/data/dataset/celeba_hq_256_noised'

os.makedirs(res_dir)

mean=0.
var=np.random.randint(225,3025)

img_num=len(img_dir)

for i in range(img_num):
    img=cv2.imread(os.path.join('/data/dataset/celeba_hq_256',img_dir[i]))
    
    noise=np.random.normal(mean,var**0.5,(256,256,3))
    img=img+noise
    cv2.imwrite(os.path.join(res_dir,img_dir[i]),img)

print(img_num)

