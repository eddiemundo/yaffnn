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
    
    All in all this class is like a very bad version of NumPy without the extra
    cool numerical numpy stuff although I might add those later.
    
    Some ambiguity, sometimes I don't consider 1x1 matrices to be matrices but
    allow them because they might be intermediate steps in some chain of
    multiplications. The times I don't consider them is when using __setitem__
    if __setitem__ returns a 1x1 matrix it returns a scalar number instead
    """
    
    def __init__(self, matrix=None, rows=None, cols=None):
        """The one that rules them all"""
        # our internal representation of the matrix
        self.matrix = None
        # number of rows
        self.rows = None
        # number of columns
        self.cols = None
        # tensor order (0 order is scalar, but for this application it is
        # empty matrix
        self.order = None

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
        return self.matrix == other.matrix and self.rows == other.rows and self.cols == other.cols and self.order == other.order
    
    def __add__(self, other):
        """Overloads the + operator"""
        if self.rows == other.rows and self.cols == other.cols:
            matrix = [self.matrix[i] + other.matrix[i] for i in range(len(self.matrix))]
            
            # hack that uses knowledge of internal representation
            result = Matrix(matrix)
            result.order = self.order
            result.rows = self.rows
            result.cols = self.cols
            return result
        else:
            raise DimensionError("Dimensions of the matrices don't match.")
    
    def __sub__(self, other):
        """Overloads the - operator"""
        if self.rows == other.rows and self.cols == other.cols:
            matrix = [self.matrix[i] - other.matrix[i] for i in range(len(self.matrix))]
            
            # hack that uses knowledge of internal representation
            result = Matrix(matrix)
            result.order = self.order
            result.rows = self.rows
            result.cols = self.cols
            return result
        else:
            raise DimensionError("Dimensions of the matrices don't match")
        
    def __mul__(self, other):
        """
        Overloads the * operator. Does matrix multiplication.
        __rmul__ for scalar multiplication when the scalar is on the left
        """
        
        matrix = []
        if isNum(other):
            matrix = [self.matrix[i]*other for i in range(len(self.matrix))]
        elif isinstance(other, Matrix):
            if self.cols != other.rows:
                raise DimensionError("Number of rows of A != to number of columns of B")
            else:
                numRows = self.rows
                numCols = self.cols
                for rowNum in range(numRows):
                    row = self.matrix[rowNum * numCols:rowNum * numCols + numCols]
                    for colNum in range(numRows):
                        col = other.matrix[colNum::other.cols]
                        sum = 0
                        for i in range(numCols):
                            sum += row[i]*col[i] 
                        matrix.append(sum)
                result = Matrix(matrix)
                result.order = 2
                result.rows = self.rows
                result.cols = other.cols
                
                
                leftRows = [self.matrix[i*self.cols: i*self.cols + self.cols] for i in range(self.rows)]
                rightCols = [self.matrix[i::other.cols] for i in range(other.cols)]
                result = [zip(leftRows[i], rightCols[i]) for i in range(self.rows)]
                result = [i[0]*i[1] for item in result for i in item]
                
        return result
    
    def dot(self, other):
        if not isinstance(other, Matrix):
            raise TypeError('Right operand is not a Matrix')
        if self.rows != 1 and self.cols != 1:
            raise DimensionError('Left operand is not a vector')
        if other.rows != 1 and other.cols != 1:
            raise DimensionError('Right operand is not a vector')
        
        return sum([self.matrix[i] * other.matrix[i] for i in range(len(self.matrix))])

    
    def transpose(self):
        """Transposes current matrix"""
        matrix = []
        for i in range(self.cols):
            matrix += self.matrix[i::self.cols]
        self.matrix = matrix
        cols = self.cols
        self.cols = self.rows
        self.rows = cols
    
    def randomize(self):
        for i in range(len(self.matrix)):
            self.matrix[i] = random.random()
    
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
        self.inputVectorList = []
        self.outputVectorList = []
        self.deltaVectorList = []
        
        # each element of this list represents a weight matrix
        # each weight matrix represents the connection weights between a node layer
        self.weightMatrixList = []
        
        # initialize weight matrices, initial matrices are zeroed
        for i in range(len(topology) - 1):
            self.weightMatrixList.append(Matrix(rows=topology[i], cols=topology[i+1]))


    def activationFunction(self, input):
        """The activation function for each node"""
        None
        
    def activationFunctionDerivative(self, input):
        """The derivative of the activation function"""
        None
        
    def forwardProp(self, input):
        """
        Forward propagate input through the network, storing inputs and
        outputs for each node in the network
        """
        initialInput = Matrix(input)
        # we have dummy node layer representing the initial input
        self.inputVectorList.append(initialInput)
        self.outputVectorList.append(initialInput)
        
        numWeightLayers = len(self.topology) - 1
        currentInputList = []
        for i in range(numWeightLayers):
            for columnVector in self.weightMatrixList[i]:
                currentInputList.append(self.outputVectorList[i] * columnVector)
            currentInputVector = Matrix(currentInputList)
            self.inputVectorList.append(Matrix(currentInputVector))
            self.outputVectorList.append(self.activationFunction(Matrix(currentInputVector)))

    def backwardProp(self, target):
        """
        Backward propagate outputs through the network, storing 'deltas' for
        each node.
        TODO: Things to check:
        - that the size of the target vector matches the number of output nodes
        - that the input, output vector lists have been initialized by
        forwardProp
        """
        targetVector = Matrix(target)
        
        # calculate the initial deltas for the output layer
        self.deltaVectorList.prepend(self.outputVectorList[-1] - targetVector)
        numWeightLayers = len(self.topology) - 1
        
        for i in range(numWeightLayers):
            deltaVector = self.weightMatrixList[-1 - i] * self.deltaVectorList[-1 - i].transpose()
            deltaVector.transpose()
            self.deltaVectorList[-1 - i].transpose()
            self.deltaVectorList.prepend(deltaVector)

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
        """Matrices of the same dimensions should be added"""
        self.assertEquals(Matrix([1, 2, 3]) + Matrix([1, 2, 3]), Matrix([2, 4, 6]))
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[1, 2], [3, 4]])
        self.assertEquals(m1 + m2, Matrix([[2, 4], [6, 8]]))
        
#    def testMulMatrix(self):
#        """Matrices of compatible dimensions should be multiplied"""
#        m1 = Matrix([[1, 2, 3], [4, 5, 6]])
#        m2 = Matrix([[1, 2], [3, 4], [5, 6]])
#        m3 = m1*m2
#        self.assertEquals(m1 * m2, Matrix([[22, 28], [49, 64]]))
#        
#    def testTranspose(self):
#        m = Matrix([[1, 4], [2, 5], [3, 6]])
#        m.transpose()
#        self.assertEquals(m, Matrix([[1,2,3],[4,5,6]]))
#        m.transpose()
#        self.assertEquals(m, Matrix([[1,4],[2,5],[3,6]]))
#    def testMalformedMatrixInit(self):
#        """Malformed matrices should raise exceptions"""
#        self.assertRaises(MalformedMatrix, Matrix, [[1], 2])
#        self.assertRaises(MalformedMatrix, Matrix, [[]])
#        self.assertRaises(MalformedMatrix, Matrix, [[[1]]])
#        self.assertRaises(MalformedMatrix, Matrix, [[1], [2, 3], [4, 5]])
#
#    def testDimensionErrorAdd(self):
#        """Operations with matrices that don't match should raise exceptions"""
#        m1 = Matrix([1])
#        self.assertRaises(DimensionError, m1.__add__, Matrix([1, 2]))
#        m1 = Matrix([1, 2])
#        m2 = Matrix([1, 2])
#        self.assertRaises(DimensionError, m1.__mul__, m2)

if __name__ == "__main__":
    unittest.main()