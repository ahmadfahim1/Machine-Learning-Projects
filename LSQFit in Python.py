import numpy as np
import matplotlib.pyplot as plt

def sq_sum(x):
    return np.dot(x,x)

def s_xy(x,y):
    return np.dot(x,y)

def my_linefit(x,y):
    # Calculate N (number of data points)
    N = len(x)

    # Calculate the necessary sums
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x_squared = sq_sum(x)
    sum_xy = s_xy(x,y)

    #a = (sum_x*sum_y) - (b* sum_x)
    #b = (sum_y - a * sum_x)
    b = (sum_y / N - (sum_x * sum_xy)/(N*sum_x_squared)/(1-(sum_x ** 2/(N * sum_x_squared))))
    a = (sum_x * -b +sum_xy)/ sum_x_squared
    return a,b

x_point =[]
y_point =[]

def on_click(event):
    if event.button == 1:
        x_point.append(event.x)
        y_point.append(event.y)
        z.clear()
        z.scatter(x_point , y_point, color ='g', marker = 'o', label ='Points')
        z.set_xticks([-2,-1,0,1,2,3,4,5])
        z.set_yticks([-2,-1,0,1,2,3,4,5])
        z.set_xlabel('X-axis')
        z.set_ylabel('Y-axis')
        z.legend("linear regression")
        plt.draw()
    elif event.button == 3:
        plt.close()

fig, z =plt.subplots()
z.set_xticks([-2,-1,0,1,2,3,4,5])
z.set_yticks([-2,-1,0,1,2,3,4,5])
z.scatter(x_point , y_point, color ='g', marker = 'o', label ='Points')
fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()

fig, z = plt.subplots()
a, b = my_linefit(x_point, y_point)
zp = np.arange(-2, 5, 0.1)
z.plot(zp, a*zp + b, 'r')
z.scatter(x_point, y_point, c = 'g', marker = 'o', label = 'Point')
plt.show()

    
