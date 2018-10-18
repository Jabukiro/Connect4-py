""" Changes connect-4py to a proper format for the ANN"""


def Data():
    with open('connect-4.data', 'r') as file:
        c = file.read()
        data = c.split()

    with open('data.txt', 'w') as file:
        i = 0
        temp = ""
        length = len(data[0])
        while i in range(length):
            for j in data[i]:
                if j == ',':
                    temp +=j
                elif j == 'x':
                    temp += '1'
                elif j == 'o':
                    temp += '-1'
                elif j == 'b':
                    temp += '0'
                elif j == 'w':
                    temp +='1\n'
                    break
                elif j == 'd':
                    temp += '0\n'
                    break
                elif j == 'l':
                    temp += '-1\n'
                    break
            i +=1
            file.write(temp)
    print("finish")