import cv2
import numpy as np
import glob
import os
#设置寻找亚像素角点的参数（即寻找棋盘边角的参数）
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)     #两个参数，前者是最大循环次数，后者是最大误差容限,即循环30次或误差达到0.001以下即停止寻找

#获取标定板角点的位置
board_size=[6,9] #棋盘的格子个数or角点数？
scale=20 #应该是每格长度
objp=np.zeros((board_size[0]*board_size[1],3),np.float32) #设置一个空二维矩阵作为存储棋盘角点三维坐标的地址，size为[6*9，3]
objp[:,:2]=np.mgrid[0:(board_size[0]-1)*scale:complex(0,board_size[0]),0:(board_size[1]-1)*scale:complex(1,board_size[1])].T.reshape(-1,2)
#将三维坐标用grid的形式存储并改变矩阵格式，T为转置，reshape()为改变数组size，变为1行，2列，自动计算多少层
obj_points=[] #存储3D坐标
img_points=[] #存储左侧2D坐标
img_points_r=[] #存储右侧3D坐标

#左侧相机内参标定
images=glob.glob('picture/left*.png')  #用glob读取所有左侧相机的图片文件路径
#print(images)
for fname in images:
    img=cv2.imread(fname)
    img=cv2.resize(img,(320,240))            #读取并统一图片size
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #灰度化图片
    size=gray.shape[::-1]    #得到图片的size（就是320，240）
    ret,corners=cv2.findChessboardCorners(gray,(board_size[0],board_size[1]),None)   #找出图中角点 ret是bool值，表示是否包含亚像素角点，corners是坐标
    #print(corners)
    if ret:            #假如包含棋盘
        obj_points.append(objp)                     #把棋盘每个点的坐标导入进3D坐标里
        corners2=cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),criteria)          #在原坐标基础上寻找亚像素角点（或将亚角点坐标进一步精确化）
        #print(corners2)
        if corners.any: #(如果包含亚像素角点)
            img_points.append(corners2/1.0) #应该是用于修改精度或者数据格式
        else:
            img_points.append(corners/1.0)  
ret,mtx,dist,rvecs,tvecs=cv2.calibrateCamera(obj_points,img_points,size,None,None,flags=0)   #得到内外矩阵参数
print(mtx)
    