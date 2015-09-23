
# coding: utf-8

# In[1]:

# iterative closest point 
# inspired by http://stackoverflow.com/questions/20120384/iterative-closest-point-icp-implementation-on-python


# In[2]:

import cv2
import numpy as np
import sys


# In[3]:

def del_miss(indeces, dist, max_dist, thRate = 0.8):
    thDist = max_dist * thRate
    return np.array([indeces[0][np.where(dist.T[0] < thDist)]])

#sample
# indeces = np.array([[1,5,7, 9]])
# dist = np.array([[1, 10, 5, -1]]).astype(np.float32).T
# max_dist = 10.0
# print indeces
# print del_miss(indeces, dist, max_dist)


# In[4]:

def icp(d1, d2):
    src = np.array([d1.T], copy=True).astype(np.float32)
    dst = np.array([d2.T], copy=True).astype(np.float32)
    
    knn = cv2.KNearest()
    responses = np.array(range(len(d2[0]))).astype(np.float32)
    knn.train(src[0], responses)
        
    Tr = np.array([[np.cos(0), -np.sin(0), 0],
                   [np.sin(0), np.cos(0),  0],
                   [0,         0,          1]])

    dst = cv2.transform(dst, Tr[0:2])
    max_dist = sys.maxint
       
    for i in range(10):
        ret, results, neighbours, dist = knn.find_nearest(dst[0], 1)
        
        indeces = results.astype(np.int32).T     
        indeces = del_miss(indeces, dist, max_dist)  
        
        T = cv2.estimateRigidTransform(dst[0, indeces], src[0, indeces], True)

        max_dist = np.max(dist)
        dst = cv2.transform(dst, T)
        Tr = np.dot(np.vstack((T,[0,0,1])), Tr)        
        
    return Tr[0:2]


# In[5]:

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    x1 = np.linspace(0,1.1,100)
    y1 = np.sin(x1 * np.pi)
    d1 = np.array([x1, y1])

    th = np.pi/8
    rot = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    move = np.array([[0.15], [0.05]])
    d2 = np.dot(rot, d1) + move

    ret = icp(d1,d2)

    plt.plot(d1[0], d1[1])
    plt.plot(d2[0], d2[1])
    dst = np.array([d2.T], copy=True).astype(np.float32)
    dst = cv2.transform(dst, ret)
    plt.plot(dst[0].T[0], dst[0].T[1])
    plt.show()

    print ret[0][0]*ret[0][0]+ret[0][1]*ret[0][1]
    print np.arccos(ret[0][0]) /2/np.pi*360
    print np.arcsin(ret[0][1]) /2/np.pi*360

    print ret


# In[ ]:



