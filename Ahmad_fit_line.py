import numpy as np
import matplotlib.pyplot as plt


def my_linefit(x, y):
    
    N = len(x)

    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x_squared = np.dot(x, x)
    sum_xy = np.dot(x, y)

    a_numerator = N * sum_xy - sum_x * sum_y
    a_denominator = N * sum_x_squared - sum_x**2
    a = a_numerator / a_denominator

    b = (sum_y - a * sum_x) / N
    return a, b



x_point = []
y_point = []

def on_click(event):
    if event.button == 1:
        x_point.append(event.xdata)  
        y_point.append(event.ydata)  
        z.clear()
        z.scatter(x_point, y_point, color='g', marker='o', label='Points')
        z.set_xlabel('X-axis')
        z.set_ylabel('Y-axis')
        z.legend(["linear regression"])  
        plt.draw()
    elif event.button == 3:
        plt.close()

fig, z = plt.subplots()
z.scatter(x_point, y_point, color='g', marker='o', label='Points')
z.set_xticks([-5,-4,-3,-2, -1, 0, 1, 2, 3, 4, 5])
z.set_yticks([-5,-4,-3,-2, -1, 0, 1, 2, 3, 4, 5])
fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()

fig, z = plt.subplots()
a, b = my_linefit(x_point, y_point)
zp = np.arange(-5, 5, .01)
z.plot(zp, a * zp + b, '-r')
z.scatter(x_point, y_point, c='g', marker='o', label='Point')
plt.xticks([-5,-4,-3,-2, -1, 0, 1, 2, 3, 4, 5])  
plt.yticks([-5,-4,-3,-2, -1, 0, 1, 2, 3, 4, 5])  
#plt.legend(["Fitted Line", "Points"])  # Update the legend labels
plt.show()
