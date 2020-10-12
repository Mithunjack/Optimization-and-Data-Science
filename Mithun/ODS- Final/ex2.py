import numpy as np
import matplotlib.pyplot as plt

def n_big_coef(ck, n):
	ck1 = np.absolute(ck)
	print('ck1',ck1)
	#ck2 = np.argsort(ck1)[:n]
	#print(ck2)
	ck1_sort = (np.sort(ck1))[-n:]
	# print('ck1_sort',ck1_sort)

	ck1 = list(ck1)
	x = np.zeros(len(ck), dtype= 'complex128')
	# print('x: ',x)
	for i in range(n):
		pos = ck1.index(ck1_sort[i])
		# print(pos)
		x[pos] = ck[pos]
		print(x[pos])
		#x[i] = ck[i]
	# print('zero value: ',x)
	return x

#def min_appx(apx, dax)
def ifft_ncoeff (arr, n):
    return np.fft.irfft(n_big_coef(arr,n))

def best_approximation(avg_absolute_deff,ck,dax_data):
	counter = 1
	while counter < ck.size:
	 data = np.fft.irfft(n_big_coef(ck,counter))
	 print('data:' , data)
	 diff =	np.average(np.absolute(data - dax_data))
	 print(diff)
	 if diff > avg_absolute_deff:
		 counter = counter + 1
	 else:
		 return counter


 
if __name__ == '__main__':	
	dax = np.genfromtxt('dax_data.txt',delimiter = ' ')
	dax = np.array(dax)
	print("dax data: ",dax.size)

	daxx = np.fft.rfft(dax)
	print("daxx data: ",daxx.size)
	# dax1 = np.fft.irfft(n_big_coef(daxx, 1))
	# print('dax1 data: ',dax1)
	dax5 = np.fft.irfft(n_big_coef(daxx, 5))
	# print('dax5 data: ',dax5)

	# dax10 = np.fft.irfft(n_big_coef(daxx, 10))
	# dax123 = np.fft.irfft(n_big_coef(daxx, 123))
	# ```````````````````````````
	print('AVG dax data: ',np.average(dax))

	# 1b
	print(best_approximation(100,daxx,dax))
	#
	# plt.plot(dax, 'r')
	# plt.plot(dax1, 'y')
	# plt.plot(dax5, 'g')
	# plt.plot(dax10, 'b')
	# plt.plot(dax123, 'b')
	# plt.plot(dax5, 'b', dax1, 'r', dax10)
	# plt.xticks(rotation=70)
	# plt.yticks(rotation=70)
	# plt.show()
	#

