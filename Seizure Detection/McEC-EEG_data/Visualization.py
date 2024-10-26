import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from numpy import array
import matplotlib.patches as patches

# 创建一个图和一个坐标轴


points_dict={'O1': array([360, 678]),
 'F7': array([218, 240]),
 'A1': array([ 42, 390]),
 'A2': array([858, 390]),
 'T5': array([218, 574]),
 'Pz': array([450, 552]),
 'T4': array([736, 406]),
 'F3': array([334, 262]),
 'O2': array([540, 678]),
 'P4': array([566, 552]),
 'T3': array([164, 406]),
 'F4': array([566, 262]),
 'C4': array([592, 406]),
 'C3': array([306, 406]),
 'T6': array([680, 572]),
 'Fp2': array([542, 138]),
 'Fp1': array([358, 138]),
 'P3': array([334, 552]),
 'Fz': array([450, 262]),
 'F8': array([680, 240]),
 'Cz': array([450, 408]),
 }

def draw_circles_at_indexes(ax, indexes, radius=35, edgecolor='#8B0000'):
    for index in indexes:
        if index in points_dict:
            point = points_dict[index]
            # circle = patches.Circle(point, radius=radius, edgecolor=edgecolor, facecolor=(1, 0.5, 0.5,0.55))
            circle = patches.Circle(point, radius=radius, edgecolor=edgecolor, facecolor='#8B000070')
            ax.add_patch(circle)
    ax.axis('off')



# 载入图像数据
# Importing the patches module from matplotlib

# Now we will redefine the function to draw multiple circles based on a list of indexes


# We will create a new figure and axis
fig, ax = plt.subplots()


# Load the image data again
image = plt.imread('nd.png')
ax.imshow(image)
columns = ['no','Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2','F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'Fz', 'Cz', 'Pz']

# 读取CSV文件
df = pd.read_csv('3_Resultdata.csv', header=None)

# 使用提供的列名
df.columns = columns

# 创建一个空的列表来存储需要打印红圈的索引
example_indexes = []

hangshu=70

# 遍历第一行的值，如果为1，则将对应的索引添加到example_indexes列表中
for i, val in enumerate(df.iloc[hangshu+1]):
    if val == 1 and columns[i] != 'no':
        example_indexes.append(columns[i])









# example_indexes = ['Pz']  # This list can be changed to whichever indexes the user chooses
#
#
# # Draw red circles at the specified indexes
draw_circles_at_indexes(ax, example_indexes)
#
# # Display the result
plt.show()
# plt.savefig(f'{hangshu}.png')