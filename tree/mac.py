import numpy as np

VDD = 1.8
PERIOD = 2.6
prob = 0.9
length = 100
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
            input_sequence.append(np.random.choice([0, 1], p=[0.7, 0.3]))
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
    result.append(str(hex(np.sum(temp)))[2:].upper().zfill(2))


for i in range(length):
    string = "".join(str(x) for x in array_t[i])
    input_hex.append(str(hex(int(string, base=2)))[2:].upper().zfill(size))

# print(input_hex)
# print (result)

with open('out.txt', 'w') as file:
    if(size == 16):
        file.write(";add \" .VEC 'input.vec' \" to your .sp file to include the vector file\n")
        file.write(";Vector Pattern\n")
        file.write("Radix 4444444444444444" + "\n")
        file.write("\t+4444444444444444\n")
        file.write("\t+24\n")
        file.write("vname in<[63:60]> in<[59:56]> in<[55:52]> in<[51:48]> in<[47:44]> in<[43:40]> in<[39:36]> in<[35:32]>\n \
        +in<[31:28]> in<[27:24]> in<[23:20]> in<[19:16]> in<[15:12]> in<[11:8]> in<[7:4]> in<[3:0]>\n \
        +in2<[63:60]> in2<[59:56]> in2<[55:52]> in2<[51:48]> in2<[47:44]> in2<[43:40]> in2<[39:36]> in2<[35:32]>\n \
        +in2<[31:28]> in2<[27:24]> in2<[23:20]> in2<[19:16]> in2<[15:12]> in2<[11:8]> in2<[7:4]> in2<[3:0]>\n \
        +result<[5:4]> result<[3:0]>\n")
        file.write("io    iiiiiiiiiiiiiiii\n \
        +iiiiiiiiiiiiiiii\n \
        +oo\n\n")
    elif(size == 8):
        file.write(";add \" .VEC 'input.vec' \" to your .sp file to include the vector file\n")
        file.write(";Vector Pattern\n")
        file.write("Radix 44444444" + "\n")
        file.write("\t+44444444\n")
        file.write("\t+24\n")
        file.write("vname in<[31:28]> in<[27:24]> in<[23:20]> in<[19:16]> in<[15:12]> in<[11:8]> in<[7:4]> in<[3:0]>\n \
        +in2<[31:28]> in2<[27:24]> in2<[23:20]> in2<[19:16]> in2<[15:12]> in2<[11:8]> in2<[7:4]> in2<[3:0]>\n \
        +result<[5:4]> result<[3:0]>\n")
        file.write("io    iiiiiiii\n \
        +iiiiiiii\n \
        +oo\n\n")
    elif(size == 4):
        file.write(";add \" .VEC 'input.vec' \" to your .sp file to include the vector file\n")
        file.write(";Vector Pattern\n")
        file.write("Radix 4444" + "\n")
        file.write("\t+4444\n")
        file.write("\t+14\n")
        file.write("vname in<[15:12]> in<[11:8]> in<[7:4]> in<[3:0]>\n \
        +in2<[15:12]> in2<[11:8]> in2<[7:4]> in2<[3:0]>\n \
        +result<[4:4]> result<[3:0]>\n")
        file.write("io    iiii\n \
        +iiii\n \
        +oo\n\n")
        
    file.write("Tunit ns\n")
    file.write("Period " + str(PERIOD) + "\n")
    file.write("Odelay 1.8\n")
    file.write("Trise 0.1\n")
    file.write("Tfall 0.1\n")
    file.write("VIH " + str(VDD) + "\n")
    file.write("VIL 0\n")
    file.write("VOH " + str(VDD*0.9) + "\n")
    file.write("VOL " + str(VDD*0.1) + "\n\n")
    
    
    for i in range(length):
        if(size == 16):
            file.write(input_hex[i] + ' ' + f16 + ' ' + result[i] + '\n' )
        elif(size == 8):
            file.write(input_hex[i] + ' ' + f8 + ' ' + result[i] + '\n' )
        elif(size == 4):
            file.write(input_hex[i] + ' ' + f4 + ' ' + result[i] + '\n' )