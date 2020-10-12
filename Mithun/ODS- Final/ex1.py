import numpy as np
from numpy import linalg as LA
#
# a = np.array( [[1,2],
# 			  [2,4]] )
# b = np.array( [[2,3],
# 				[3,4]])
#
# print(np.matmul(a,b))

# Excercise 1.1 => 3b
array_1 = np.array([0,-1,-2])
array_2 = np.array([1,-3])
array_3 = np.array([1,-3])
array_4 = np.array([1,-3])
array_5 = np.array([1,2])
array_6 = np.array([1,-2])

x = np.dot(np.transpose(array_3),array_4)
y = np.dot(array_5,np.transpose(array_6))
xy = np.dot(x,y)

infinit_norm = LA.norm(array_1, np.inf)
euclid_norm = LA.norm(array_2,2)
resultHalf = np.dot((infinit_norm**2),(euclid_norm**-2))
result = np.dot(resultHalf,xy)
print(result);
#
# # In[3]:
#
#
# x = 3 + 3j
# y = 4 + 5j
# print(x+y)
#
#
# # In[4]:
#
#
# # exp(2)
#
#
# # In[5]:
#
#
# np.exp(2)
#
#
# # In[6]:
#
#
# # print(2i + 4i)
#
#
# # In[7]:
#
#
# print(2j + 4j)
#
#
# # In[8]:
#
#
# print(a*2)
#
#
# # In[9]:
#
#
# z= a*2
#
#
# # In[10]:
#
#
# print()
#
#
# # In[11]:
#
#
# print(z)
#
#
# # In[12]:
#
#
# print(a)
#
#
# # In[13]:
#
#
# a1 = a
#
#
# # In[14]:
#
#
# print(a)
#
#
# # In[15]:
#
#
# print(a1)
#
#
# # In[16]:
#
#
# a1 = np.array([1,-3])
# a2 = np.array([[1], [-3]])
# print(np.matmul(a1,a2))
#
#
# # In[17]:
#
#
# print(max
#      )
#
#
# # In[18]:
#
#
# print(np.max)
#
# # In[27]:
#
#
# print(np.finfo(np.float64).max)
#
#
# # In[28]:
#
#
# print(np.finfo(np.float32).max)
#
#
# # In[29]:
#
#
# print(np.finfo(np.float64).tiny)
#
#
# # In[30]:
#
#
# print(np.finfo(np.float32).max)
#
#
# # In[31]:
#
#
# print(np.finfo(np.float64).precision)
#
#
# # In[32]:
#
#
# print(np.finfo(np.float32).precision)
#
#
# # In[33]:
#
#
# print(np.finfo(np.double).precision)
#
#
# # In[34]:
#
#
# print(np.finfo(np.double).tiny)
#
#
# # In[35]:
#
#
# print(np.finfo(np.float).max)
#
#
# # In[36]:
#
#
# print(np.finfo(np.float64).max)
#
#
# # In[37]:
#
#
# print(np.finfo(np.double).max)
#
#
# # In[44]:
#
#
# from numpy import linalg
# m0 = np.array([[0],[-1],[-2]])
# linalg.norm(m0, np.inf)
#
#
# # In[60]:
#
#
# from numpy import linalg as LA
# m0 = np.array([1,-3,2])
# m1 = np.array([[1],[-1]])
# m1 = m1.transpose()
# m2 = np.array([[1,2,0],
#               [-3, -2 ,1]])
# m3 = np.array([1,-3,2])
# res = LA.norm(m0, 1) + np.matmul(m1,np.matmul(m2,m3))
# print(res)
#
#
# # In[84]:
#
#
#
# m0 = np.array([0,-1,-2])
# m1 = np.array([1,-3])
# m2 = np.array([[1],[-3]])
# m3 = np.array([[1],[-3]])
# m4 = np.array([[1],[2]])
# m5 = np.array([[1],[-2]])
# res = np.power(LA.norm(m0, np.inf) ,2) * np.power(LA.norm(m1, 2), -2) * np.multiply(np.matmul(m2.transpose(),m3), np.matmul(m4,m5.transpose()) )
# print(res)
#
# # @ can be used instead of np.matmul




