import numpy as np

prob = 0.9
length = 50
size = 16
f16= "FFFFFFFFFFFFFFFF"
f8 = "FFFFFFFF"
f4 = "FFFF"

array = []
array_t = []
input_hex = []
result = []

for i in range(size*4):
    input_sequence = []
    for j in range(length):
        if(j == 0):
            input_sequence.append(np.random.choice([0, 1], p=[0.5, 0.5]))
        else:
            if(np.random.rand() < prob):
                if(input_sequence[j-1] == 1):
                    input_sequence.append(0)
                else:
                    input_sequence.append(1)
            else:
                input_sequence.append(input_sequence[j-1])
    array.append(input_sequence)

for i in range(length): 
    temp = []
    for j in range(size*4):
        temp.append(array[j][i])
    array_t.append(temp)
    result.append(str(hex(np.sum(temp)))[2:].upper())


for i in range(length):
    string = "".join(str(x) for x in array_t[i])
    input_hex.append(str(hex(int(string, base=2)))[2:].upper().zfill(size))

# print(input_hex)
# print (result)

with open('out.txt', 'w') as file:
    for i in range(length):
        if(size == 16):
            file.write(input_hex[i] + ' ' + f16 + ' ' + result[i] + '\n' )
        elif(size == 8):
            file.write(input_hex[i] + ' ' + f8 + ' ' + result[i] + '\n' )
        elif(size == 4):
            file.write(input_hex[i] + ' ' + f4 + ' ' + result[i] + '\n' )