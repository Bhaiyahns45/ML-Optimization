# importing libraries
import numpy as np
import time
import matplotlib.pyplot as plt

# creating initial data values
# of x and y
x = [1,2,3,4,5,6,7,8,9,10]
y = [10,9,8,7,6,5,3,1,1,0]
# to run GUI event loop

# here we are creating sub plots

plt.title("Geeks For Geeks", fontsize=20)

# setting x-axis label and y-axis label
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

x=1
y=10
# Loop
for _ in range(10):
	# creating new Y values

    x+=1
    y-=1


    plt.plot(x,y)

    time.sleep(0.1)
    plt.show()
