import unittest
import random
import math

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
class InvalidInput(Exception):
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
    
    self.order is currently worthless
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
                raise InvalidInput("Rx0 or 0xC matrix")
            elif rows < 0 or cols < 0:
                raise InvalidInput("rows or cols is negative")
            elif rows == 1 or cols == 1:
                self.order = 1
            elif rows > 1 and cols > 1:
                self.order = 2
            else:
                self.order = 0
            self.rows = rows
            self.cols = cols
            self.matrix = []
            for i in range(rows * cols):
                self.matrix.append(0)
        elif matrix != None and rows == None and cols == None:
            # these if statements are order dependent
            if not isinstance(matrix, list):
                raise InvalidInput("Matrix is not a list")
            elif matrix == []:
                self.rows = 0
                self.cols = 0
                self.order = 0
                self.matrix = matrix
            elif all([isinstance(row, list) for row in matrix]):
                rowLengths = [len(row) for row in matrix]
                if rowLengths[0] == 0:
                    raise InvalidInput("Row has zero length")
                if rowLengths.count(rowLengths[0]) != len(rowLengths):
                    raise NotRectangular("Row lengths don't match")
                if not all([isNum(element) for row in matrix for element in row]):
                    raise InvalidInput("Row elements are not numbers")
                self.rows = len(rowLengths)
                self.cols = rowLengths[0]
                if self.rows == 1 or self.cols == 1:
                    self.order = 1
                else:
                    self.order = 2
                self.matrix = [element for row in matrix for element in row]
            elif all([isNum(element) for element in matrix]):
                self.rows = 1
                self.cols = len(matrix)
                self.order = 1
                self.matrix = matrix
            else:
                raise InvalidInput("Some rows in matrix are not lists")
        elif matrix == None and rows == None and cols == None:
            self.rows = 0
            self.cols = 0
            self.order = 0
            self.matrix = []
        else:
            raise InvalidInput("Either matrix is given or rows and cols are given, not both.")

    def __getitem__(self, key):
        """
        Handles [1,1], [slice, 1], [1, slice], [slice, slice], [1], [slice]
        Don't try slices with negative indices or negative indices in general
        """
        colSlice = slice(None, None, None)
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
                colSlice = slice(key[1], key[1] + 1)
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
        # calculate the slices representing the rows of the matrix
        rowSlices = [slice(self.cols * rowIndex, self.cols * rowIndex + self.cols) for rowIndex in rowIndices]
        # select only the chosen columns from these row slices
        resultMatrix = [self.matrix[rowSlice][colSlice] for rowSlice in rowSlices]
        # if the result is a 1x1 matrix then return a scalar
        if len(resultMatrix[0]) == 1 and len(resultMatrix) == 1:
            return resultMatrix[0][0]
        # else return a Matrix
        else:
            return Matrix(resultMatrix)

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
            result = Matrix([self.matrix[i]*other for i in range(len(self.matrix))])
            result.rows = self.rows
            result.cols = self.cols
            return result
        elif isinstance(other, Matrix):
            if self.cols != other.rows:
                raise DimensionError("Number of rows of A != to number of columns of B")
            else:
                leftRows = [self.matrix[i*self.cols: i*self.cols + self.cols] for i in range(self.rows)]
                
                rightCols = [other.matrix[i::other.cols] for i in range(other.cols)]
                result = [self._dot(leftRows[i], rightCols[j]) for i in range(len(leftRows)) for j in range(len(rightCols))]
                # uses internal structure of Matrix
                result = Matrix(result)
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
    
    def randomize(self):
        for i in range(len(self.matrix)):
            self.matrix[i] = random.random()

    def clone(self):
        matrix = self.matrix[:]
        result = Matrix(matrix)
        result.rows = self.rows
        result.cols = self.cols
        return result
        
    
class NeuralNetwork:
    """A feedforward neural network"""
    def __init__(self, topology):
        """Sets up relevant matrices/vectors for the neural network"""
            
        # topology = [#inputs, #hidden nodes, ..., #hidden nodes, #outputs]
        self.topology = topology

        # each element of these vectors represents a node layer in the network
        # each node layer is represented by a vector matrix
        # each element in the vector matrix represents an individual node
        # vectors in these lists are assumed as row vectors
        # we count the input and output layers as a node layer
        self.inputsList = [None for i in range(len(topology))]
        self.outputsList = [None for i in range(len(topology))]
        # note: first index of deltasList is never used
        self.deltasList = [None for i in range(len(topology))]
        
        # each element of this list represents a weight matrix
        # each weight matrix represents the connection weights between a node layer
        self.weightMatrixList = []
        # represents the corrections to the weight matrix after batch or online training
        self.weightMatrixCorrections = []
        
        # initialize weight matrices, initial matrices are zeroed
        for i in range(len(topology) - 1):
            # rows is +1 because we are adding weights for the bias too
            self.weightMatrixList.append(Matrix(rows=topology[i]+1, cols=topology[i+1]))
            self.weightMatrixList[i][:] = 1
            self.weightMatrixCorrections.append(Matrix(rows=topology[i]+1, cols=topology[i+1]))
            
    @staticmethod
    def tanh(list):
        """Takes a list, returns a list"""
        # not efficient, dont think catastrophic things can occur though
        return [(math.exp(num)-math.exp(-num))/(math.exp(num)+math.exp(-num)) for num in list]
    @staticmethod
    def tanhDerivative(list):
        """Not too useful"""
        tanhResults = NeuralNetwork.tanh(list)
        return [1 - pow(tanhResult,2) for tanhResult in tanhResults] 

        
    def forwardProp(self, input):
        """
        Forward propagate input through the network, storing inputs and
        outputs for each node in the network
        
        
        Algorithm:
        Give input layer input
        Make first output layer same as input
        send first output layer to first hidden layer through weight matrix
        record the activation
        """
        if not isinstance(input, list):
            raise TypeError('forwardProp expects a list for input')
        
        
        self.inputsList[0] = Matrix(input)
        output = input[:]
        output.append(1)
        # add the bias
        self.outputsList[0] = Matrix(output)
        
        for i in range(len(self.weightMatrixList)):
            
            self.inputsList[i+1] = self.outputsList[i]*self.weightMatrixList[i]
            # output of node layer without bias
            output = self.tanh(self.inputsList[i+1].matrix)
            # output of node layer with bias 
            output.append(1)
            self.outputsList[i+1] = Matrix(output)
        # this is important cause the output layer doesn't do anything to the
        # inputs it gets
        # we don't need to add a bias to the output 
        self.outputsList[-1] = self.inputsList[-1]


    def backwardProp(self, target):
        """
        Backward propagate outputs through the network, storing 'deltas' for
        each node.
        TODO: Things to check:
        - that the size of the target vector matches the number of output nodes
        - that the input, output vector lists have been initialized by
        forwardProp
        """
        
        
        # TODO: store the dError/dweights into a matrix of their own (or apply them
        # to the weight matrices directly if we're doing online training)
        
#        calculate the deltas for the output layer
#        for each hidden layer
#            calculate the deltas for that hidden layer using the deltas in the
#            layer ahead of it
        #    delta_j = h'(a_j) * sum(wjk * delta_k)
        
        # calculate the deltas for the output layer
        self.deltasList[-1] = self.outputsList[-1] - Matrix(target)
        for i in reversed(range(1, len(self.topology) - 1)):
            product = (self.weightMatrixList[i] * self.deltasList[i+1].tranpose()).transpose()
            # the deltas are row vectors
            self.deltasList[i] = (1-self.outputsList[i]).multiply(product)
            
        for i in reversed(range(len(self.weightMatrixList))):
            self.weightMatrixCorrections[i] = self.outputsList[i].transpose() * self.deltasList[i+1]
        
        
        # weight matrix corrections are wij = xi * delta_j
        # so
        # w00 = x0 * delta_0 w01 = x0 * delta_1
        # w10 = x1 * delta_0 w11 = x1 * delta_1
        # w20 = x2 * delta_0 w21 = x2 * delta_1
        #
        # outputs xi are all row vectors
        # deltas delta_i are all row vectors
        #
        #
        #
        
def trainOnline(nn, dataset, targets):
    """
    Takes a NeuralNetwork and a dataset, and targets lists of lists
    The dataset and targets lists should match in length and each element
    of each list should have the same length as well.
    """
    for data, target in zip(dataset, targets):
        nn.forwardProp(data)
        nn.backwardProp(target)
        for weightMatrix, correction in zip(nn.weightMatrixList, nn.weightMatrixCorrections):
            weightMatrix = weightMatrix - correction
        
    
            
        
            


class NeuralNetworkTest(unittest.TestCase):
    def test__init__(self):
        nn = NeuralNetwork([2, 2, 2])
        self.assertEquals(nn.topology, [2,2,2])
        self.assertEquals(nn.inputsList, [None, None, None])
        self.assertEquals(nn.outputsList, [None, None, None])
        m1 = Matrix(rows=3, cols=2)
        m1[:] = 1
        
        self.assertEquals(nn.weightMatrixList[0], m1)
        self.assertEquals(nn.weightMatrixList[1], m1)
        self.assertEquals(len(nn.weightMatrixList), 2)

    def testForwardProp(self):
        self.assertEquals(NeuralNetwork.tanhDerivative([1]), [0.41997434161402614])
        self.assertEquals(NeuralNetwork.tanh([1]), [0.7615941559557649])
        # apparently not like C, where their numbers are sometimes held in special cpu
        # registers. but maybe it is like that and doesn't manifest on this machine
        self.assertEquals(NeuralNetwork.tanh([2,2]), [0.964027580075817,0.964027580075817])
        nn = NeuralNetwork([2, 2, 2])
        nn.weightMatrixList[0][:] = 0
        nn.weightMatrixList[1][:] = 0
        nn.forwardProp([0,0])
        self.assertEquals(nn.outputsList, [Matrix([0,0,1]), Matrix([0,0,1]), Matrix([0,0])])
        self.assertEquals(nn.inputsList, [Matrix([0,0]), Matrix([0,0]), Matrix([0,0])])
        nn = NeuralNetwork([1,3,3])
        nn.forwardProp([1])
        
        self.assertEquals(nn.outputsList[0], Matrix([1, 1]))
        self.assertEquals(nn.outputsList[1], Matrix([0.964027580075817,0.964027580075817,0.964027580075817,1]))
        self.assertEquals(nn.outputsList[2], Matrix([3.892082740227451,3.892082740227451,3.892082740227451]))

        
        

class MatrixTest(unittest.TestCase): 
    def testMatrix(self):
        """Matrix initialized with parameters"""
        # Pass tests
        m = Matrix()
        self.assertEquals(m.order, 0)
        self.assertEquals(m.rows, 0)
        self.assertEquals(m.cols, 0)
        self.assertEquals(m.matrix, [])
        m = Matrix([])
        self.assertEquals(m.order, 0)
        self.assertEquals(m.rows, 0)
        self.assertEquals(m.cols, 0)
        self.assertEquals(m.matrix, [])
        m = Matrix([1, 2, 3])
        self.assertEquals(m.order, 1)
        self.assertEquals(m.rows, 1)
        self.assertEquals(m.cols, 3)
        self.assertEquals(m.matrix, [1, 2, 3])
        m = Matrix([[1, 2, 3]])
        self.assertEquals(m.order, 1)
        self.assertEquals(m.rows, 1)
        self.assertEquals(m.cols, 3)
        self.assertEquals(m.matrix, [1, 2, 3])
        m = Matrix([[1], [2], [3]])
        self.assertEquals(m.order, 1)
        self.assertEquals(m.rows, 3)
        self.assertEquals(m.cols, 1)
        self.assertEquals(m.matrix, [1, 2, 3])
        m = Matrix([[1, 2], [3, 4], [5, 6]])
        self.assertEquals(m.order, 2)
        self.assertEquals(m.rows, 3)
        self.assertEquals(m.cols, 2)
        self.assertEquals(m.matrix, [1, 2, 3, 4, 5, 6])
        m = Matrix(rows=0, cols=0)
        self.assertEquals(m.order, 0)
        self.assertEquals(m.rows, 0)
        self.assertEquals(m.cols, 0)
        self.assertEquals(m.matrix, [])
        m = Matrix(rows=2, cols=3)
        self.assertEquals(m.order, 2)
        self.assertEquals(m.rows, 2)
        self.assertEquals(m.cols, 3)
        self.assertEquals(m.matrix, [0, 0, 0, 0, 0, 0])
        # Fail tests
        self.assertRaises(InvalidInput, Matrix, [[]])
        self.assertRaises(InvalidInput, Matrix, [[], []])
        self.assertRaises(InvalidInput, Matrix, [[[], []], [[], []]])
        self.assertRaises(InvalidInput, Matrix, 'a')
        self.assertRaises(InvalidInput, Matrix, [1, [2]])
        self.assertRaises(NotRectangular, Matrix, [[1], [2, 3]])
        self.assertRaises(NotRectangular, Matrix, [[1, 1], [2, 2], [3]])
        args = []
        kwargs = {'rows':1, 'cols':0}
        self.assertRaises(InvalidInput, Matrix, *args, **kwargs)
        args = [[]]
        kwargs = {'rows':1, 'cols':1}
        self.assertRaises(InvalidInput, Matrix, *args, **kwargs)
        args = [[]]
        kwargs = {'rows':1}
        self.assertRaises(InvalidInput, Matrix, *args, **kwargs)
        args = [[]]
        kwargs = {'cols':1}
        self.assertRaises(InvalidInput, Matrix, *args, **kwargs)
        args = []
        kwargs = {'rows':1}
        self.assertRaises(InvalidInput, Matrix, *args, **kwargs)
        args = []
        kwargs = {'cols':1}
        self.assertRaises(InvalidInput, Matrix, *args, **kwargs)
        args = []
        kwargs = {'cols':-1, 'rows':1}
        self.assertRaises(InvalidInput, Matrix, *args, **kwargs)

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
        
        self.assertEquals(m[1,1], 5)
        self.assertEquals(m[1,:], Matrix([4,5,6]))
    
        self.assertEquals(m[:,1], Matrix([[2],[5],[8]]))
        self.assertEquals(m[0:1,1], Matrix([[2],[5]]))
        self.assertEquals(m[::1,1], Matrix([[2],[5],[8]]))
        self.assertEquals(m[:,1:2], Matrix([[2,3],[5,6],[8,9]]))
        # Fail
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


if __name__ == "__main__":
    unittest.main()