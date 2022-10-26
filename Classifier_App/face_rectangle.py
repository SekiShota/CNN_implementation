#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Dlibで顔部分を切り抜き
import cv2, dlib, sys, glob, pprint
import matplotlib.pyplot as plt

#Dlibを始める
detector=dlib.get_frontal_face_detector()

#入力画像をリサイズするか
flag_resize=False

#顔画像を取得して保存
def get_face_rect(fname):
    global fid
    img=cv2.imread(fname)

    #サイズが大きければリサイズ
    if flag_resize:
        img=cv2.resize(img, None, fx=0.2, fy=0.2,)

    #顔検出
    dets=detector(img,1)
    print(dets)
    for k,d in enumerate(dets):
        pprint.pprint(d)
        x1=int(d.left())
        y1=int(d.top())
        x2=int(d.right())
        y2=int(d.bottom())
        im=img[y1:y2, x1:x2]
        try:
            im=cv2.resize(im, (64,64))
        except:
            continue
#         cv2.rectangle(img, (x1, y1), (x2, y2), color=(0,0,255), thickness=4)
        
    return im,d

