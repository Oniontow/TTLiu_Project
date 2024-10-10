import numpy as np
import matplotlib.pyplot as plt
import csv

type = 'TT, wa=.60'
voltage = '1.8V'

with open('VQOUT1 vs VQ1.csv', 'r') as curve1:
    reader = csv.reader(curve1)
    data1 = list(reader)
for i in range(7):
    data1.pop(0)
data1.pop(-1)
    
with open('VQOUT2 vs VQ2.csv', 'r') as curve2:
    reader = csv.reader(curve2)
    data2 = list(reader)
for i in range(7):
    data2.pop(0)
data2.pop(-1)
    
x1 = []
y1 = []
x2 = []
y2 = []

for i in range(len(data1)):
    x1.append(float(data1[i][0].split('E')[0]) * (10 ** int(data1[i][0].split('E')[1])))
    y1.append(float(data1[i][1].split('E')[0]) * (10 ** int(data1[i][1].split('E')[1])))
for i in range(len(data2)):
    y2.append(float(data2[i][0].split('E')[0]) * (10 ** int(data2[i][0].split('E')[1])))
    x2.append(float(data2[i][1].split('E')[0]) * (10 ** int(data2[i][1].split('E')[1])))

plt.plot(x1, y1)
plt.plot(x2, y2)
plt.title('Vout vs Vin, '+type+', '+voltage)
plt.xlabel('Vin1, Vout2')
plt.ylabel('Vout1, Vin2')
plt.savefig('figure/Vout vs Vin, '+type+', '+voltage +'.png')
plt.show()

matrix1 = np.array([x1, y1])
matrix2 = np.array([x2, y2])
rotation_matrix = np.array([[1/np.sqrt(2), -1/np.sqrt(2)], [1/np.sqrt(2), 1/np.sqrt(2)]])
new_matrix1 = np.matmul(rotation_matrix, matrix1)
new_matrix2 = np.matmul(rotation_matrix, matrix2)

start = 0
end = len(new_matrix2[0])-1
while (start < end) :
    new_matrix2[0][start], new_matrix2[0][end] = new_matrix2[0][end], new_matrix2[0][start]
    new_matrix2[1][start], new_matrix2[1][end] = new_matrix2[1][end], new_matrix2[1][start]
    start = start + 1
    end = end - 1

plt.plot(new_matrix1[0], new_matrix1[1])
plt.plot(new_matrix2[0], new_matrix2[1])
plt.title('Vout vs Vin in u-v axis, '+type+', '+voltage)
plt.xlabel('u')
plt.ylabel('v')
plt.savefig('figure/Vout vs Vin in u-v axis, '+type+', '+voltage+'.png')
plt.show()

difference = [[],[]]

minimal_tick = np.fabs(x1[1] - x1[0])

for i in range(len(new_matrix1[0])):
    for j in range(len(new_matrix2[0])):
        if new_matrix1[0][i] <= new_matrix2[0][j]+0.5*minimal_tick and new_matrix1[0][i] >= new_matrix2[0][j]-0.5*minimal_tick:
            difference[0].append(new_matrix1[0][i])
            difference[1].append(new_matrix1[1][i] - new_matrix2[1][j])

#for SNM, RNM
NM = min(np.fabs(max(difference[1])), np.fabs(min(difference[1])))
#end of SNM, RNM

# #for WNM
# local_max = 0
# NM = 1/np.sqrt(2)
# for i in range(1, len(difference[0])-1):
#     if(difference[1][i] < difference[1][i-1] and difference[1][i] < difference[1][i+1]):
#         NM *= difference[1][i]
#         local_max = 1
#         break
# if local_max == 0:
#     NM *= max(difference[1])
# #end of WNM
    
NM = str(NM)
NM = NM[0:5]
plt.plot(difference[0], difference[1])
plt.title('Difference, NM = '+ NM+', '+type+', '+voltage)
plt.savefig('figure/Difference, NM = '+ NM+', '+type+', '+voltage+'.png')
plt.show()