import numpy as np

def matrixLoader(directory,version,length):
    fn = directory + '/' +'matrix_' + str(version) + '-' + str(length) + '.txt'
    
    with open(fn,'r') as f:
        array = []
        num = 0
        for line in f:
            if num == 0:
                size = line.split()
            else:
                array.append(float(line))
            num += 1
        A = np.array(array)
        
        if version =='EFG':
            size = int(size[0]) + int(size[1])
        else:
            size = int(size[0])
        
        A = np.reshape(A,(size,size))
        return A
        
        
def vectorLoader(directory,version,length):
    fn = directory + '/' +'vector_' + str(version) + '-' + str(length) + '.txt'
    
    with open(fn,'r') as f:
        array = []
        num = 0
        for line in f:
            if num == 0:
                pass
            else:
                array.append(float(line))
            num += 1
        b = np.array(array)
        return b