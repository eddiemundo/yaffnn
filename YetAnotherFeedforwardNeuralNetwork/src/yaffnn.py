import unittest
import random
import math
import cProfile
import pstats

class MalformedMatrix(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return str(self.msg)
class DimensionError(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return str(self.msg)
class InputError(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return str(self.msg)
class NotRectangular(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return str(self.msg)

def isNum(s):
    """Tests if s is a number"""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False

class Matrix:
    """
    The idea behind this class is that you give the constructor a list, or
    list of lists representing a vector or matrix, and it gives you an object
    representing a vector or matrix with all the operations associated with
    these sorts of objects. It does NOT take block matrices though.
    
    All in all this class is like a very bad version of NumPy/matlab without the
    extra cool numerical numpy/matlab stuff although I might add those later.
    
    Note ambiguity: sometimes scalars are returned instead of Matrix when doing
    some operations like dot product. Operations on 0x0 matrices are not allowed
    
    The method named "multiply" is an elementwise multiplication
    
    You can use slice notation on the matrices but don't try descending slices.
    Probably are bugs with that cause I don't care about reversing matrices
    right now
    
    SPEED: Probably very slow, maybe turn this into C later
    
    TODO: Clean up exception handling by putting it all at the beginning of method
    when possible
    Look over features and methods for efficiency and clean up code in general
    """
    
    def __init__(self, matrix=None, rows=None, cols=None):
        """The one that rules them all"""
        # our internal representation of the matrix
        self.matrix = None
        # number of rows
        self.rows = None
        # number of columns
        self.cols = None

        if matrix == None and rows != None and cols != None:
            if (rows == 0 and cols != 0) or (rows != 0 and cols == 0):
                raise InputError("Rx0 or 0xC matrix")
            elif rows < 0 or cols < 0:
                raise InputError("rows or cols is negative")

            self.rows = rows
            self.cols = cols
            self.matrix = [0 for i in range(rows * cols)]
            
        elif matrix != None and rows == None and cols == None:
            # these if statements are order dependent
            if not isinstance(matrix, list):
                raise InputError('Matrix is not a list')
            elif matrix == []:
                self.rows = 0
                self.cols = 0
                self.matrix = []
            elif all([isinstance(row, list) for row in matrix]):
                rowLengths = [len(row) for row in matrix]
                if rowLengths[0] == 0:
                    raise InputError("Row has zero length")
                elif rowLengths.count(rowLengths[0]) != len(rowLengths):
                    raise NotRectangular("Row lengths don't match")
                elif not all([isNum(element) for row in matrix for element in row]):
                    raise InputError("Row elements are not numbers")
                self.rows = len(rowLengths)
                self.cols = rowLengths[0]
                self.matrix = [element for row in matrix for element in row]
            elif all([isNum(element) for element in matrix]):
                self.rows = 1
                self.cols = len(matrix)
                self.matrix = matrix[:]
            else:
                raise InputError("Some rows in matrix are not lists")
        elif matrix == None and rows == None and cols == None:
            self.rows = 0
            self.cols = 0
            self.matrix = []
        else:
            raise InputError("Either matrix is given or rows and cols are given, not both.")

    def __getitem__(self, key):
        """
        Handles [1,1], [slice, 1], [1, slice], [slice, slice], [1], [slice]
        Don't try slices with negative indices or negative indices in general
        
        
        This method takes either a number/slice, or a 2-tuple containing numbers/slices
        """
        if isNum(key):
            numCols = self.cols
            start = key * numCols
            
            result = Matrix()
            result.matrix = self.matrix[start: start + numCols]
            result.rows = 1
            result.cols = numCols
            return result
        
        elif isinstance(key, slice):
            numCols = self.cols
            resultMatrix = []
            # convert raw slice to indices
            keyIndices = key.indices(self.rows)
            # 1:5 means 1 to 5 inclusive
            rowIndices = range(*slice(keyIndices[0], keyIndices[1] + 1, keyIndices[2]).indices(self.rows))
            # loop overhead, maybe can do better with itemgetter
            for rowNum in rowIndices:
                rowIndex = rowNum * numCols
                resultMatrix += self.matrix[rowIndex: rowIndex + numCols]
            
            result = Matrix()
            result.matrix = resultMatrix
            
            result.rows = len(rowIndices)
            result.cols = numCols
            return result
            
        elif isinstance(key, tuple):
            if len(key) == 2:
                # local variables for speed
                key0IsNum = isNum(key[0])
                key0IsSlice = isinstance(key[0], slice)
                key1IsNum = isNum(key[1])
                key1IsSlice = isinstance(key[1], slice)
                
                # using recursion because i'm lazy
                if key0IsNum and key1IsNum:
                    rowMatrix = self.__getitem__(key[0])
                    return rowMatrix.matrix[key[1]]
                elif key0IsNum and key1IsSlice:
                    rowMatrix = self.__getitem__(key[0])
                    # convert raw slice to indices
                    keyIndices = key[1].indices(rowMatrix.cols)
                    resultMatrix = rowMatrix.matrix[slice(keyIndices[0], keyIndices[1] + 1, keyIndices[2])]
                    
                    result = Matrix()
                    result.matrix = resultMatrix
                    result.rows = 1
                    # the purpose of this is to get the number of resulting columns
                    colIndices = range(*slice(keyIndices[0], keyIndices[1] + 1, keyIndices[2]).indices(rowMatrix.cols))
                    result.cols = len(colIndices)
                    return result
                elif key0IsSlice and key1IsNum:
                    rowMatrix = self.__getitem__(key[0])
        
                    result = Matrix()
                    result.matrix = rowMatrix.matrix[key[1]::self.cols]
                    result.rows = rowMatrix.rows
                    result.cols = 1
                    if len(result.matrix) == 1:
                        return result.matrix[0]
                    else:
                        return result
                elif key0IsSlice and key1IsSlice:
                    rowMatrix = self.__getitem__(key[0])
                    # slow but easy
                    resultMatrix = []
                    for i in range(rowMatrix.rows):
                        resultMatrix += rowMatrix[i, key[1]].matrix
                    
                    result = Matrix()
                    result.matrix = resultMatrix
                    result.rows = rowMatrix.rows
                    result.cols = len(result.matrix)/rowMatrix.rows
                    if len(result.matrix) == 1:
                        return result.matrix[0]
                    else:
                        return result
                else:
                    raise TypeError('Elements of key tuple are not numbers or slices')
            else:
                raise TypeError('Key tuple is not a 2-tuple')
        else:
            raise TypeError('Key is not a number, slice, or tuple')
        

    def __setitem__(self, key, value):
        """
        Handles [1,1], [slice, 1], [1, slice], [slice, slice], [1], [slice]
        Don't try slices with negative indices or negative indices in general
        """
        colSlice = slice(0, self.cols, 1)
        if isNum(key):
            rowIndices = [key]
        elif isinstance(key, slice):
            indexSlice = key
            if indexSlice.start == None:
                start = 0
            elif isNum(indexSlice.start):
                start = indexSlice.start
            else:
                raise TypeError('Slice start is not a number')
            if indexSlice.stop == None:
                # watch out when copy paste row for rows col for cols
                stop = self.rows
            elif isNum(indexSlice.stop):
                stop = indexSlice.stop + 1
            else:
                raise TypeError('Slice stop is not a number')
            if indexSlice.step == None:
                step = 1
            elif isNum(indexSlice.step):
                step = indexSlice.step
            else:
                raise TypeError('Slice step is not a number')
            rowIndices = range(start, stop, step)
        elif isinstance(key, tuple) and len(key) == 2:
            # handle the first argument in the tuple
            if isNum(key[0]):
                rowIndices = [key[0]]
            elif isinstance(key[0], slice):
                indexSlice = key[0]
                if indexSlice.start == None:
                    start = 0
                elif isNum(indexSlice.start):
                    start = indexSlice.start
                else:
                    raise TypeError('Slice start is not a number')
                if indexSlice.stop == None:
                    # watch out when copy paste row for rows col for cols
                    stop = self.rows
                elif isNum(indexSlice.stop):
                    stop = indexSlice.stop + 1
                else:
                    raise TypeError('Slice stop is not a number')
                if indexSlice.step == None:
                    step = 1
                elif isNum(indexSlice.step):
                    step = indexSlice.step
                else:
                    raise TypeError('Slice step is not a number')
                rowIndices = range(start, stop, step)
            else:
                raise TypeError('Row index is not an integer or slice')
            # handle the second argument in the tuple
            if isNum(key[1]):
                colSlice = slice(key[1], key[1] + 1, 1)
            elif isinstance(key[1], slice):
                indexSlice = key[1]
                if indexSlice.start == None:
                    start = 0
                elif isNum(indexSlice.start):
                    start = indexSlice.start
                else:
                    raise TypeError('Slice start is not a number')
                if indexSlice.stop == None:
                    # watch out when copy paste row for rows col for cols
                    stop = self.cols 
                elif isNum(indexSlice.stop):
                    stop = indexSlice.stop + 1
                else:
                    raise TypeError('Slice stop is not a number')
                if indexSlice.step == None:
                    step = 1
                elif isNum(indexSlice.step):
                    step = indexSlice.step
                else:
                    raise TypeError('Slice step is not a number')
                colSlice = slice(start, stop, step)
            else:
                raise TypeError('Column index is not an integer or slice')
        elif isinstance(key, tuple):
            raise TypeError('More than 2 arguments')
        else:
            raise TypeError('Index is not a number')
        
        # out of bounds errors for column slices
        if colSlice.stop > self.cols:
            raise IndexError('Column slice out of bounds')
        if colSlice.start < 0:
            raise IndexError('Column slice out of bounds')
        
        # calculate the slices representing the rows of the matrix with columns sliced
        rowSlices = [slice(self.cols * rowIndex + colSlice.start, self.cols * rowIndex + colSlice.stop, colSlice.step) for rowIndex in rowIndices]
        valueList = []
        numRowsToAssign = len(rowSlices)
        numColumnsToAssign = (1+math.floor(((colSlice.stop - colSlice.start)-1)/colSlice.step))
        numElementsToAssign = numRowsToAssign*numColumnsToAssign
         
        if isNum(value):
            # might be buggy for decending slices
            # only append the number of columns that need assigning
            # so if [:, 0], then only append once
            # if [:, 0:1], then append twice    
            for i in range(numElementsToAssign):
                valueList.append(value)
        elif isinstance(value, list):
            
            if len(value) == numElementsToAssign:
                valueList = value
            else:
                raise ValueError('Value list not same length as matrix entries to be assigned')
        
        valueList = [valueList[i*numColumnsToAssign:i*numColumnsToAssign + numColumnsToAssign] for i in range(numRowsToAssign)]
        # select only the chosen columns from these row slices and assign values
        for i,rowSlice in enumerate(rowSlices):
            self.matrix[rowSlice] = valueList[i]
        

    def __eq__(self, other):
        """Allows == comparisons"""
        return self.matrix == other.matrix and self.rows == other.rows and self.cols == other.cols
    
    def __add__(self, other):
        """Returns a new Matrix representing a scalar or Matrix added to this Matrix"""
        if not (isNum(other) or isinstance(other, Matrix)):
            raise TypeError("Operand is not a Matrix or number")
        # uses short circuit logic
        if isinstance(other, Matrix) and (self.rows != other.rows or self.cols != other.cols):
            raise DimensionError("Dimensions of the matrices don't match")

        thisMatrix = self.matrix
        if isNum(other):
            resultMatrix = [element + other for element in thisMatrix]
        else:
            otherMatrix = other.matrix
            resultMatrix = [thisElement + otherMatrix[otherIndex] for otherIndex, thisElement in enumerate(thisMatrix)]
        
        # uses internal structure of Matrix
        result = Matrix()
        result.matrix = resultMatrix
        result.rows = self.rows
        result.cols = self.cols
        return result
    
    def __radd__(self, other):
        return self.__add__(other)
        
    
    def __sub__(self, other):
        """Returns a new Matrix representing a scalar or Matrix subtracted from this Matrix"""
        if not (isNum(other) or isinstance(other, Matrix)):
            raise TypeError("Operand is not a Matrix or number")
        # uses short circuit logic
        if isinstance(other, Matrix) and (self.rows != other.rows or self.cols != other.cols):
            raise DimensionError("Dimensions of the matrices don't match")

        thisMatrix = self.matrix
        if isNum(other):
            resultMatrix = [element - other for element in thisMatrix]
        else:
            otherMatrix = other.matrix
            resultMatrix = [thisElement - otherMatrix[otherIndex] for otherIndex, thisElement in enumerate(thisMatrix)]
        
        # uses internal structure of Matrix
        result = Matrix()
        result.matrix = resultMatrix
        result.rows = self.rows
        result.cols = self.cols
        return result
    
    def __rsub__(self, other):
        """Can't just call __sub__ because subtraction is not commutative"""
        if not (isNum(other) or isinstance(other, Matrix)):
            raise TypeError("Operand is not a Matrix or number")
        # uses short circuit logic
        if isinstance(other, Matrix) and (self.rows != other.rows or self.cols != other.cols):
            raise DimensionError("Dimensions of the matrices don't match")

        thisMatrix = self.matrix
        if isNum(other):
            resultMatrix = [other - element for element in thisMatrix]
        else:
            otherMatrix = other.matrix
            resultMatrix = [otherMatrix[otherIndex] - thisElement for otherIndex, thisElement in enumerate(thisMatrix)]
        
        # uses internal structure of Matrix
        result = Matrix()
        result.matrix = resultMatrix
        result.rows = self.rows
        result.cols = self.cols
        return result
        
    def __mul__(self, other):
        """
        Overloads the * operator. Does matrix multiplication.
        __rmul__ for scalar multiplication when the scalar is on the left
        """
        if isNum(other):
            result = Matrix()
            result.matrix = [element * other for element in self.matrix]
            result.rows = self.rows
            result.cols = self.cols
            return result
        elif isinstance(other, Matrix):
            if self.cols != other.rows:
                raise DimensionError("Number of rows of A != to number of columns of B")
            else:
                leftRows = [self.matrix[i*self.cols: i*self.cols + self.cols] for i in range(self.rows)]
                rightCols = [other.matrix[i::other.cols] for i in range(other.cols)]
                
                resultMatrix = [sum([row[i] * col[i] for i in range(len(row))]) for row in leftRows for col in rightCols]

                result = Matrix()
                result.matrix = resultMatrix
                result.rows = self.rows
                result.cols = other.cols
                return result
    
    def __rmul__(self, other):
        """
        Used for scalar multiplication when a scalar is the left operand
        """
        return self.__mul__(other)
    
    def __str__(self):
        """
        Prints out the matrix, not too nice looking since it uses a space between
        each number
        """
        result = "[\n"
        for i in range(self.rows):
            for j in range(self.cols):
                result += " " + str(self.matrix[i * self.cols + j])
            result += "\n"
        result += "]"
        return result
    
    def _dot(self, v1, v2):
        """
        v1 and v2 are arrays of the same length
        returns dot product of them.
        This is a private helper method used in __mul__
        Probably get rid of this later
        """
        return sum([v1[i] * v2[i] for i in range(len(v1))])
    
    def multiply(self, other):
        """
        Elementwise subtraction for Matrix 's only.
        """
        if not isinstance(other, Matrix):
            raise TypeError("Operand is not a Matrix")
        elif self.rows != other.rows or self.cols != other.cols:
            raise DimensionError("Dimensions of matrices don't match")
        
        matrices = zip(self.matrix, other.matrix)
        resultMatrix = [this * other for this, other in matrices]
        # uses internal structure of Matrix
        result = Matrix()
        result.matrix = resultMatrix
        result.rows = self.rows
        result.cols = self.cols
        return result
    
    def dot(self, other):
        """
        Dot products vectors
        """
        if not isinstance(other, Matrix):
            raise TypeError('Right operand is not a Matrix')
        if self.rows != 1 and self.cols != 1:
            raise DimensionError('Left operand is not a vector')
        if other.rows != 1 and other.cols != 1:
            raise DimensionError('Right operand is not a vector')
        
        return sum([self.matrix[i] * other.matrix[i] for i in range(len(self.matrix))])

    
    def transpose(self):
        """Return a Matrix that is the transpose of self"""
        cols = [self.matrix[i::self.cols] for i in range(self.cols)]
        transpose = []
        for col in cols:
            transpose.append(col)
        
        return Matrix(transpose)
    
    def pow(self, num):
        """
        Returns new Matrix representing elementwise exponentiation to the
        power of num of this Matrix's elements
        TODO: Input checking
        """
        resultMatrix = [pow(element, 2) for element in self.matrix[:]]
        result = Matrix()
        result.matrix = resultMatrix
        result.rows = self.rows
        result.cols = self.cols
        return result
    
    def norm(self):
        return math.sqrt(sum([pow(element,2) for element in self.matrix]))
    
    def randomize(self):
        """Randomizes the entries of the matrix to non-zero numbers between -1 and 1"""
        for i in range(len(self.matrix)):
            x = 0
            while x == 0:
                x = random.uniform(-1,1)
            self.matrix[i] = x

    def clone(self):
        matrix = self.matrix[:]
        result = Matrix(matrix)
        result.rows = self.rows
        result.cols = self.cols
        return result
        
    
class NeuralNetwork:
    """
    A feed-forward neural network.
    
    TODO:
    1. Try a graph data structure instead of matrices
    """
    def __init__(self, topology):
        """
        Given a topology, sets up relevant variables for the neural network
        
        Input: a list/sequence with this structure [#inputs #hiddenunits ... #outputs]
        where each element is a 'node layer' in the network
        """
        # self.topology is coupled with self.numNodeLayers and self.numWeightLayers
        self.topology = topology[:]
        
        self.numNodeLayers = len(topology)
        self.inputs = [None for i in range(self.numNodeLayers)]
        self.outputs = [None for i in range(self.numNodeLayers)]
        self.deltas = [None for i in range(self.numNodeLayers)]
        self.augmentedOutputs = [None for i in range(self.numNodeLayers)]
        
        self.numWeightLayers = len(topology) - 1
        self.weightMatrices = [Matrix(rows=topology[i]+1, cols=topology[i+1])
                               for i in range(self.numWeightLayers)]
        # initialize to random non-zero values in [-1,1] 
        for weightMatrix in self.weightMatrices:
            weightMatrix.randomize()
        self.weightCorrections = [Matrix(rows=topology[i]+1, cols=topology[i+1])
                                  for i in range(self.numWeightLayers)]

    @staticmethod
    def tanh(input):
        """Returns the tanh of input"""
        if not isNum(input) and not isinstance(input, list):
            raise TypeError("Input is not a number or list")
        if not isNum(input) and isinstance(input, list):
            if input == []:
                raise TypeError("Input list has no elements")
            elif any([not isNum(element) for element in input]):
                raise TypeError("Input list has non-number elements")
        
        if isNum(input):
            # math.exp can't underflow
            return math.tanh(input)
#            try:
#                exppos = math.exp(input)
#                expneg = math.exp(-input)
#            except OverflowError:
#                if input > 0:
#                    return 1
#                else:
#                    return -1
#            return (exppos - expneg)/(exppos + expneg)
        else:
#            result = []
#            for element in input:
#                try:
#                    exppos = math.exp(element)
#                    expneg = math.exp(-element)
#                    result.append((exppos - expneg)/(exppos + expneg))
#                except OverflowError:
#                    if element > 0:
#                        result.append(1)
#                    else:
#                        result.append(-1)
                
            #return result
            return [math.tanh(element) for element in input]

    def forwardProp(self, input):
        """
        Forward propagate input through the network, storing inputs, outputs
        for each node in the network for the back propagation phase
        """
        if not isinstance(input, list):
            raise TypeError('Input is not a list')
        else:
            if input == []:
                raise TypeError("Input list has no elements")
            elif any([not isNum(element) for element in input]):
                raise TypeError("Input list has non-number elements")

        self.inputs[0] = Matrix(input)
        self.outputs[0] = self.inputs[0].clone()
        output = input[:]
        output.append(1)
        self.augmentedOutputs[0] = Matrix(output)

        for i in range(1, self.numNodeLayers - 1):
            self.inputs[i] = self.augmentedOutputs[i-1] * self.weightMatrices[i-1]
            
            # apply the activation function to the inputs
            outputs = self.tanh(self.inputs[i].matrix)
            self.outputs[i] = Matrix(outputs)
            # does not affect self.outputs[i] cause Matrix copies outputs
            outputs.append(1)
            self.augmentedOutputs[i] = Matrix(outputs)
        
        self.inputs[-1] = self.augmentedOutputs[-2] * self.weightMatrices[-1]
        self.outputs[-1] = self.inputs[-1].clone()
        
        return self.outputs[-1].matrix[:]

    def backwardProp(self, target):
        """
        Backward propagate outputs through the network storing 'deltas' for
        each node (except input nodes).
        """
        if not isinstance(target, list):
            raise TypeError('Target is not a list')
        else:
            if target == []:
                raise TypeError("Target list has no elements")
            elif any([not isNum(element) for element in target]):
                raise TypeError("Target list has non-number elements")
        
        self.deltas[-1] = self.outputs[-1] - Matrix(target)
        for i in reversed(range(1, self.numNodeLayers - 1)):
            self.deltas[i] = (self.weightMatrices[i][:self.topology[i]-1] * self.deltas[i+1].transpose()).transpose().multiply(1-self.outputs[i].pow(2))
            
        for i in range(self.numWeightLayers):
            self.weightCorrections[i] = self.augmentedOutputs[i].transpose() * self.deltas[i+1]

    def trainOnline(self, inputset, targets, learningRate, tolerance):
        """
        Takes a topology, a dataset, and a targets lists of lists
        The dataset and targets lists should match in length and each element
        of each list should have the same length as well.
        eg, dataset = [[1, 1], [2, 2]], targets = [1, 2], topology=[2,2,1]
        
        
        E = 1/2 * sum[(yk-tk)^2]
        """
        
        #change to max float someday
        errorSum = 9999999
        epochNumber = 0
        
        while errorSum > tolerance:
            errorSum = 0
            
            for input, target in zip(inputset, targets):
                result = self.forwardProp(input)
                self.backwardProp(target)
            
                error = Matrix(result) - Matrix(target)
                errorSum = errorSum + sum([pow(element, 2) for element in error.matrix])/2
            
                #online weight correction - hard coded learning rate 0.1
                for i in range(self.numWeightLayers):
                    self.weightMatrices[i] -= learningRate * self.weightCorrections[i]
            
            epochNumber += 1

            

        print('----------------')
        print('Error: ' + str(errorSum))
        print('Number of epoches: ' + str(epochNumber))                

    

class NeuralNetworkTest(unittest.TestCase):
    def test__init__(self):
        nn = NeuralNetwork([1, 2, 1])
        self.assertEquals(nn.topology, [1, 2, 1])
        self.assertEquals(nn.inputs, [None, None, None])
        self.assertEquals(nn.outputs, [None, None, None])
        self.assertEquals(nn.augmentedOutputs, [None, None, None])
        self.assertEquals(nn.deltas, [None, None, None])
        self.assertEquals(len(nn.weightMatrices), 2)
        self.assertEquals(nn.weightMatrices[0].rows, 2)
        self.assertEquals(nn.weightMatrices[0].cols, 2)
        self.assertEquals(nn.weightMatrices[1].rows, 3)
        self.assertEquals(nn.weightMatrices[1].cols, 1)

#    def testTanh(self):
#        self.assertEquals(NeuralNetwork.tanh(1), 0.7615941559557649)
#        self.assertEquals(NeuralNetwork.tanh([1]), [0.7615941559557649])
#        self.assertEquals(NeuralNetwork.tanh(2), 0.964027580075817)
#        self.assertEquals(NeuralNetwork.tanh([2,2]), [0.964027580075817,0.964027580075817])
#        self.assertEquals(NeuralNetwork.tanh(710), 1)
#        self.assertEquals(NeuralNetwork.tanh(-710), -1)

    def testForwardProp(self):
        """Tests forwardProp()"""
        
        # apparently not like C, where their numbers are sometimes held in cpu
        # registers. but maybe it is like that and doesn't manifest on this machine
        
        nn = NeuralNetwork([1, 2, 1])
        result = nn.forwardProp([0])
        self.assertEquals(nn.inputs[0], nn.outputs[0])
        self.assertEquals(NeuralNetwork.tanh(nn.inputs[1].matrix), nn.outputs[1].matrix)
        self.assertEquals(nn.inputs[2], nn.outputs[2])
        self.assertEquals(nn.outputs[2].matrix, result)
        
    def testBackwardProp(self):
        """Sees if network can be trained to approx a linear function"""
        nn = NeuralNetwork([1, 2, 1]) 
        errorSum = 1
        epochNumber = 0
        correction0 = nn.weightCorrections[0]
        correction1 = nn.weightCorrections[1]
        while errorSum > 0.0000000000000000001:
            errorSum = 0
            correction0[:] = 0
            correction1[:] = 0
            
            result = nn.forwardProp([1])
            nn.backwardProp([1])
            
            error = Matrix(result) - Matrix([1])
            errorSum = errorSum + sum([pow(element, 2) for element in error.matrix])/2
            
            #online
            nn.weightMatrices[0] = nn.weightMatrices[0] - 0.1*nn.weightCorrections[0]
            nn.weightMatrices[1] = nn.weightMatrices[1] - 0.1*nn.weightCorrections[1]
            
#            correction0 = correction0 + 0.1*nn.weightCorrections[0]
#            correction1 = correction1 + 0.1*nn.weightCorrections[1]
            
            result = nn.forwardProp([2])
            nn.backwardProp([2])
            
            error = Matrix(result) - Matrix([2])
            errorSum = errorSum + sum([pow(element, 2) for element in error.matrix])/2
            
            #online
            nn.weightMatrices[0] = nn.weightMatrices[0] - 0.1*nn.weightCorrections[0]
            nn.weightMatrices[1] = nn.weightMatrices[1] - 0.1*nn.weightCorrections[1]
#            
#            
#            
#            correction0 = correction0 + nn.weightCorrections[0]
#            correction1 = correction1 + nn.weightCorrections[1]
            
            epochNumber += 1
            
            print('Error: ' + str(errorSum))
            
#            nn.weightMatrices[0] = nn.weightMatrices[0] - correction0
#            nn.weightMatrices[1] = nn.weightMatrices[1] - correction1
        print('----------------')
        print('Number of epoches: ' + str(epochNumber))
        print('End of testBackwardProp')
#        result = nn.forwardProp([0])
#        nn.backwardProp([0])
#        
#        print('Error:')
#        error = Matrix(result) - Matrix([0])
#        print(sum([pow(element, 2) for element in error.matrix])/2)
#        print('Result')
#        print(result)
        
        
        
        
        self.assertEquals(nn.weightMatrices[0].rows, nn.weightCorrections[0].rows)
        self.assertEquals(nn.weightMatrices[0].cols, nn.weightCorrections[0].cols)
        self.assertEquals(nn.weightMatrices[1].rows,nn.weightCorrections[1].rows)
        self.assertEquals(nn.weightMatrices[1].cols, nn.weightCorrections[1].cols)
        

    def testTrainOnline(self):
        #,[3],[4],[5],[6],[7],[8],[9],[10]
        dataset = [[0,0],[0,1],[1,0],[1,1]]
        targets = [[0],[1],[1],[0]]
        nn = NeuralNetwork([2,3,1])
        nn.trainOnline(dataset, targets, 0.02, 0.00000000001)
        print(nn.forwardProp([0,0]))
        print(nn.forwardProp([1,1]))
        print(nn.forwardProp([0,1]))
        print(nn.forwardProp([1,0]))

class MatrixTest(unittest.TestCase):
    """TODO: Test exceptions"""
    def testMatrix(self):
        """Matrix initialized with parameters"""
        # Pass tests
        m = Matrix()
        self.assertEquals(m.rows, 0)
        self.assertEquals(m.cols, 0)
        self.assertEquals(m.matrix, [])
        m = Matrix([])
        self.assertEquals(m.rows, 0)
        self.assertEquals(m.cols, 0)
        self.assertEquals(m.matrix, [])
        m = Matrix([1, 2, 3])
        self.assertEquals(m.rows, 1)
        self.assertEquals(m.cols, 3)
        self.assertEquals(m.matrix, [1, 2, 3])
        m = Matrix([[1, 2, 3]])
        self.assertEquals(m.rows, 1)
        self.assertEquals(m.cols, 3)
        self.assertEquals(m.matrix, [1, 2, 3])
        m = Matrix([[1], [2], [3]])
        self.assertEquals(m.rows, 3)
        self.assertEquals(m.cols, 1)
        self.assertEquals(m.matrix, [1, 2, 3])
        m = Matrix([[1, 2], [3, 4], [5, 6]])
        self.assertEquals(m.rows, 3)
        self.assertEquals(m.cols, 2)
        self.assertEquals(m.matrix, [1, 2, 3, 4, 5, 6])
        m = Matrix(rows=0, cols=0)
        self.assertEquals(m.rows, 0)
        self.assertEquals(m.cols, 0)
        self.assertEquals(m.matrix, [])
        m = Matrix(rows=2, cols=3)
        self.assertEquals(m.rows, 2)
        self.assertEquals(m.cols, 3)
        self.assertEquals(m.matrix, [0, 0, 0, 0, 0, 0])
        # Fail tests
        self.assertRaises(InputError, Matrix, [[]])
        self.assertRaises(InputError, Matrix, [[], []])
        self.assertRaises(InputError, Matrix, [[[], []], [[], []]])
        self.assertRaises(InputError, Matrix, 'a')
        self.assertRaises(InputError, Matrix, [1, [2]])
        self.assertRaises(NotRectangular, Matrix, [[1], [2, 3]])
        self.assertRaises(NotRectangular, Matrix, [[1, 1], [2, 2], [3]])
        args = []
        kwargs = {'rows':1, 'cols':0}
        self.assertRaises(InputError, Matrix, *args, **kwargs)
        args = [[]]
        kwargs = {'rows':1, 'cols':1}
        self.assertRaises(InputError, Matrix, *args, **kwargs)
        args = [[]]
        kwargs = {'rows':1}
        self.assertRaises(InputError, Matrix, *args, **kwargs)
        args = [[]]
        kwargs = {'cols':1}
        self.assertRaises(InputError, Matrix, *args, **kwargs)
        args = []
        kwargs = {'rows':1}
        self.assertRaises(InputError, Matrix, *args, **kwargs)
        args = []
        kwargs = {'cols':1}
        self.assertRaises(InputError, Matrix, *args, **kwargs)
        args = []
        kwargs = {'cols':-1, 'rows':1}
        self.assertRaises(InputError, Matrix, *args, **kwargs)

    def test__eq__(self):
        # Pass
        self.assertEquals(Matrix(), Matrix())
        self.assertEquals(Matrix(), Matrix([]))
        self.assertEquals(Matrix(), Matrix(rows=0, cols=0))
        self.assertEquals(Matrix([]), Matrix([]))
        self.assertEquals(Matrix([]), Matrix())
        self.assertEquals(Matrix([]), Matrix(rows=0, cols=0))
        self.assertEquals(Matrix(rows=0, cols=0), Matrix(rows=0, cols=0))
        self.assertEquals(Matrix(rows=0, cols=0), Matrix())
        self.assertEquals(Matrix(rows=0, cols=0), Matrix([]))
        m = Matrix(rows=3, cols=2)
        m.matrix = [1,2,3,4,5,6]
        self.assertEquals(m, Matrix([[1, 2], [3, 4], [5, 6]]))
        # Fail
        self.assertNotEquals(Matrix([1, 2]), Matrix([1,3]))

    def test__getitem__(self):
        """
        Does not test negative indices and step sizes. I don't know if we're
        ready for that kind of pain. Probably won't test/implement negative
        or funky slices, indexing until I decide I need it.
        Doesn't test exceptions though I have exceptions in place for many errors
        """
        # Pass
        m = Matrix([[1,2,3],[4,5,6],[7,8,9]])
        self.assertEquals(m[1], Matrix([4,5,6]))
        self.assertEquals(m[1,1], 5)
        self.assertEquals(m[1,1:2], Matrix([5,6]))
        self.assertEquals(m[1:2, 1], Matrix([[5],[8]]))

        self.assertEquals(m[1:2:2, 1], 5)
        self.assertEquals(m[1:2, 1:2], Matrix([[5,6],[8,9]]))
        self.assertEquals(m[1:2:2, 1:1:2], 5)
        self.assertEquals(m[0:2:2, 0:2:2], Matrix([[1,3],[7,9]]))
        self.assertEquals(m[:], m)
        self.assertEquals(m[::2], Matrix([[1,2,3],[7,8,9]]))
        self.assertEquals(m[::3], Matrix([1,2,3]))
        self.assertEquals(m[1:2], Matrix([[4,5,6],[7,8,9]]))
        

        self.assertEquals(m[1,:], Matrix([4,5,6]))
    
        self.assertEquals(m[:,1], Matrix([[2],[5],[8]]))
        self.assertEquals(m[0:1,1], Matrix([[2],[5]]))
        self.assertEquals(m[::1,1], Matrix([[2],[5],[8]]))
        self.assertEquals(m[:,1:2], Matrix([[2,3],[5,6],[8,9]]))
#        # Fail
        self.assertNotEquals(m[1], Matrix([3,4,5]))
        self.assertNotEquals(m[0:2:2, 0:2:2], Matrix([[1,3],[7,8]]))
        
    def test__setitem__(self):
        """
        Doesn't test negative indices or slices
        Doesn't test exceptions though I have exceptions in place for many errors
        """
        # Pass
        m = Matrix(rows=2, cols=3)
        m[0,0] = 1
        self.assertEquals(m, Matrix([[1,0,0],[0,0,0]]))
        m[1] = [0,0,1]
        self.assertEquals(m, Matrix([[1,0,0],[0,0,1]]))
        m[1,:] = [0,1,0]
        self.assertEquals(m, Matrix([[1,0,0],[0,1,0]]))
        m[1] = 1
        self.assertEquals(m, Matrix([[1,0,0],[1,1,1]]))
        m[1,:] = 2
        self.assertEquals(m, Matrix([[1,0,0],[2,2,2]]))
        m[0] = 1
        self.assertEquals(m, Matrix([[1,1,1],[2,2,2]]))
        m[0,:] = 2
        self.assertEquals(m, Matrix([[2,2,2],[2,2,2]]))
        m[1,2] = 3
        self.assertEquals(m, Matrix([[2,2,2],[2,2,3]]))
        m[:,0] = 4
        self.assertEquals(m, Matrix([[4,2,2],[4,2,3]]))
        m[:,1] = [5,6]
        self.assertEquals(m, Matrix([[4,5,2],[4,6,3]]))
        m[:,1:2] = 7
        self.assertEquals(m, Matrix([[4,7,7],[4,7,7]]))
        m[:,1:2] = [1,2,3,4]
        self.assertEquals(m, Matrix([[4,1,2],[4,3,4]]))
        m[1,0::2] = [1,2]
        self.assertEquals(m, Matrix([[4,1,2],[1,3,2]]))
        m = Matrix(rows=3, cols=3)
        m[0::2,0::2] = [1,2,3,4]
        self.assertEquals(m, Matrix([[1,0,2],[0,0,0],[3,0,4]]))
        m[0::2,0::2] = 5
        self.assertEquals(m, Matrix([[5,0,5],[0,0,0],[5,0,5]]))
        self.assertEquals(m[:,:], m)
        self.assertEquals(m[:], m)
#        # Fail
#        args = [(0,0), 'a']
#        self.assertRaises(TypeError, m.__setitem__, *args)
#        args = [(0,0,0), 1]
#        self.assertRaises(TypeError, m.__setitem__, *args)
#        args = [(0,0), [1,2]]
#        self.assertRaises(TypeError, m.__setitem__, *args)
#        args = [(slice(0,3,None),0), [1,2,3,4]]
#        self.assertRaises(TypeError, m.__setitem__, *args)
#        args = [(slice(0,3,None),0), [1,2]]
#        self.assertRaises(TypeError, m.__setitem__, *args)
#        args = [('a',0), 1]
#        self.assertRaises(TypeError, m.__setitem__, *args)
#        args = [(0,'a'), 1]
#        self.assertRaises(TypeError, m.__setitem__, *args)


    def test__add__(self):
        """Tests __radd__ too"""
        self.assertEquals(Matrix([1, 2, 3]) + Matrix([1, 2, 3]), Matrix([2, 4, 6]))
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[1, 2], [3, 4]])
        self.assertEquals(m1 + m2, Matrix([[2, 4], [6, 8]]))
        m1 = Matrix([[1,2],[3,4],[5,6]])
        self.assertEquals(m1 + 1, Matrix([[2,3],[4,5],[6,7]]))
        self.assertEquals(1 + m1, Matrix([[2,3],[4,5],[6,7]]))
        
    def test__sub__(self):
        """Tests __rsub__ too"""
        self.assertEquals(Matrix([1, 2, 3]) - Matrix([1, 2, 3]), Matrix([0,0,0]))
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[1, 2], [3, 4]])
        self.assertEquals(m1 - m2, Matrix([[0, 0], [0, 0]]))
        m1 = Matrix([[1,2],[3,4],[5,6]])
        self.assertEquals(m1 - 1, Matrix([[0,1],[2,3],[4,5]]))
        self.assertEquals(1 - m1, Matrix([[0,-1],[-2,-3],[-4,-5]]))
    
    def testMultiply(self):
        """Tests elementwise multiply()"""
        self.assertEquals(Matrix([1,2,3]).multiply(Matrix([1,2,3])), Matrix([1,4,9]))
        self.assertEquals(Matrix([[1,2,3],[3,2,1]]).multiply(Matrix([[1,2,3],[3,2,1]])),Matrix([[1,4,9],[9,4,1]]))

    def test_dot(self):
        m1 = Matrix([1,2])
        m2 = Matrix([[1],[2]])
        m3 = Matrix([[1,2],[3,4]])
        # Pass
        self.assertEquals(m1.dot(m1), 5)
        self.assertEquals(m1.dot(m2), 5)
        # Fail
        self.assertRaises(DimensionError, m3.dot, m1)
        self.assertRaises(DimensionError, m1.dot, m3)
        
    
    def test__mul__(self):
        """
        Tests __rmul__ too
        """
        # Pass
        m1 = Matrix([[1, 2, 3], [4, 5, 6]])
        m2 = Matrix([[1, 2], [3, 4], [5, 6]])
        self.assertEquals(m1 * m2, Matrix([[22, 28], [49, 64]]))
        self.assertEquals(m1 * 2, Matrix([[2,4,6],[8,10,12]]))
        self.assertEquals(2 * m1, Matrix([[2,4,6],[8,10,12]]))
        
        
        m1 = Matrix([0, 0, 0])
        m2 = Matrix([[0,0],[0,0],[0,0]])
        
        self.assertEquals(m1*m2, Matrix([0,0]))
        
        m1 = Matrix([[1],[2]])
        m2 = Matrix([3,4])
        self.assertEquals(m1*m2, Matrix([[3,4],[6,8]]))
        # Fail
        self.assertRaises(DimensionError, m1.__mul__, m1)

    def testTranspose(self):
        m = Matrix([[1, 4], [2, 5], [3, 6]])
        self.assertEquals(m.transpose(), Matrix([[1,2,3],[4,5,6]]))
        self.assertEquals(m.transpose().transpose(), m)

    def test__str__(self):
        m = Matrix([[1,4], [2, 5], [3, 6]])
        self.assertEquals(m.__str__(), "[\n 1 4\n 2 5\n 3 6\n]")

cProfile.run('main()', 'inyoface')
p = pstats.Stats('inyoface')
p.strip_dirs().sort_stats('time').print_stats(20)
if __name__ == "__main__":
    
    unittest.main()
    
    