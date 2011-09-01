import unittest
import random

class MalformedMatrix(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return str(self.msg)
class MismatchingDimensions(Exception):
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

class Matrix:
    """The idea behind this class is that you give the constructor a list, or
    list of lists representing a 1D or 2D matrix, and it gives you an object
    representing a 1D or 2D matrix with all the operations associated with
    these sorts of matrices. It does NOT take block matrices though.
    
    All in all this class is like a very bad version of NumPy.
    """
    def _isNum(self, s):
        """Tests if s is a number"""
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _is1d(self, matrix):
        """Tests if the list given represents a '1d matrix'"""
        if all(map(self._isNum, matrix)):
            return True
        else:
            for row in matrix:
                # has to be a list
                if not isinstance(row, list):
                    return False
                # a list with 1 element
                if not len(row) == 1:
                    return False
                # where the element is a number
                if not self._isNum(row[0]):
                    return False
            return True
        
    def _is2d(self, matrix):
        """Tests if the list given represents a '2d matrix'"""
        # can fail if given an empty matrix
        # shouldn't happen cause empty matrix is tested beforehand
        # will fix later...
        
        # record the length of the row
        numCols = len(matrix[0])
        for row in matrix:
            # if row is not a list
            if not isinstance(row, list):
                return False
            # if row is empty
            if row == []:
                return False
            
            for element in row:
                # if element in row is not a number
                if not self._isNum(element):
                    return False
            # if the length of every row is the same as every other row
            if len(row) != numCols:
                return False
        return True
    
    def __init__(self, matrix=None, rows=None, cols=None):
        """The one that rules them all"""
        # our internal representation of the matrix
        self.matrix = []
        # number of rows
        self.rows = 0
        # number of columns
        self.cols = 0
        # tensor order (0 order is scalar, but for this application it is
        # empty matrix
        self.order = 0

        if matrix == None and rows != None and cols != None:
            if (rows == 0 and cols != 0) or (rows != 0 and cols == 0):
                raise InvalidInput("Rx0 or 0xC matrix")
            elif rows < 0 or cols < 0:
                raise InvalidInput("rows or cols is negative")
            elif rows == 1 or cols == 1:
                self.order = 1
            elif rows > 1 and cols > 1:
                self.order = 2
            # else rows=0, cols=0 and order = 0 (default values)
            self.rows = rows
            self.cols = cols
        elif matrix != None and rows == None and cols == None:
            None
        else:
            raise InvalidInput("Either matrix is given or rows and cols are given, not both.")
#        if matrix == None:
#            if rows != None and cols != None:
#                self.rows = rows
#                self.cols = cols
#                for i in range(rows*cols):
#                    self.matrix.append(0)
#            else:
#                self.order = 0
#                self.rows = 0
#                self.cols = 0
#        elif(self._is1d(matrix)):        
#            self.order = 1
#            if isinstance(matrix[0], list):
#                self.rows = len(matrix)
#                self.cols = 1
#                for row in matrix:
#                    self.matrix += row
#            else:
#                self.rows = 1
#                self.cols = len(matrix)
#                self.matrix = matrix[:] 
#        elif(self._is2d(matrix)):
#            self.order = 2
#            self.rows = len(matrix)
#            self.cols = len(matrix[0])
#            for row in matrix:
#                self.matrix += row
#        else:
#            raise MalformedMatrix("Matrix has invalid dimensions or is a block matrix.")

    def __getitem__(self, key):
        """Overloads the [] operator"""
        if self.order == 1:
            return self.matrix[key]
        elif self.order == 2:
            indexStart = self.cols * key
            return Matrix(self.matrix[indexStart:indexStart+self.cols])
        
    def __eq__(self, other):
        """Allows == comparisons"""
        return self.matrix == other.matrix and self.rows == other.rows and self.cols == other.cols and self.order == other.order
    
    def __add__(self, other):
        """Overloads the + operator"""
        if self.rows == other.rows and self.cols == other.cols:
            matrix = []
            for i in range(len(self.matrix)):
                matrix.append(self.matrix[i] + other.matrix[i])
            # hack that uses knowledge of internal representation
            result = Matrix(matrix)
            result.order = self.order
            result.rows = self.rows
            result.cols = self.cols
            return result
        else:
            raise MismatchingDimensions("Dimensions of the matrices don't match.")
    
    def __sub__(self, other):
        """Overloads the - operator"""
        if self.rows == other.rows and self.cols == other.cols:
            matrix = []
            for i in range(len(self.matrix)):
                matrix.append(self.matrix[i] - other.matrix[i])
            result = Matrix(matrix)
            result.order = self.order
            result.rows = self.rows
            result.cols = self.cols
            return result
        else:
            raise MismatchingDimensions("Dimensions of the matrices don't match")
        
    def __mul__(self, other):
        """Overloads the * operator. Does matrix multiplication... I'll decide
        whether scalar multiplication can work in here"""
        if not self.cols == other.rows:
            raise MismatchingDimensions("Rows of A not equal to Cols of B")
    
        matrix = []
        length = self.rows
        for rownum in range(length):
            row = self.matrix[rownum*self.cols:rownum*self.cols+self.cols]
            for colnum in range(length):
                col = other.matrix[colnum::other.cols]
                sum = 0
                for i in range(self.cols):
                    sum += row[i]*col[i] 
                matrix.append(sum)
        result = Matrix(matrix)
        result.order = 2
        result.rows = self.rows
        result.cols = other.cols
        return result

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
    def testEmptyMatrix(self):
        """Matrix initialized with no parameters, or empty list"""
        #Pass 
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
        #Fail
        m = Matrix([[]])
        self.assertRaises(InvalidInput, Matrix, [[]])
        m = Matrix([[], []])
        self.assertRaises(InvalidInput, Matrix, [[]])
        m = Matrix([[[], []], [[], []]])
        self.assertRaises(InvalidInput, Matrix, [[]])
        
    def testMatrix(self):
        """Matrix initialized with parameters"""
        # Pass tests
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
        self.assertRaises(InvalidInput, Matrix, 'a')
        self.assertRaises(InvalidInput, Matrix, [1, [2]])
        self.assertRaises(NotRectangular, Matrix, [[1], [2, 3]])
        self.assertRaises(NotRectangular, Matrix, [[1, 1], [2, 2], [3]])
        args = ()
        kwargs = {'rows':1, 'cols':0}
        self.assertRaises(InvalidInput, Matrix, *args, **kwargs)
        args = ([])
        kwargs = {'rows':1, 'cols':1}
        self.assertRaises(InvalidInput, Matrix, *args, **kwargs)
        args = ([])
        kwargs = {'rows':1}
        self.assertRaises(InvalidInput, Matrix, *args, **kwargs)
        args = ([])
        kwargs = {'cols':1}
        self.assertRaises(InvalidInput, Matrix, *args, **kwargs)
        args = ()
        kwargs = {'rows':1}
        self.assertRaises(InvalidInput, Matrix, *args, **kwargs)
        args = ()
        kwargs = {'cols':1}
        self.assertRaises(InvalidInput, Matrix, *args, **kwargs)
        args = ()
        kwargs = {'cols':-1, 'rows':1}
        self.assertRaises(InvalidInput, Matrix, *args, **kwargs)

#    def testAssignment(self):
#        """Assign arbitrary numbers for elements"""
#        m = Matrix(row=2, col=2)
#        m[0][0] = 1
#        m[1][1] = 1
#        self.assertEquals(m[0][0], 1)
#        self.assertEquals(m[1][1], 1)
#        self.assertEquals(m[0][1], 0)
#        self.assertEquals(m[1][0], 0)
#        self.assertEquals(m, Matrix([[1,0], [0,1]]))
#        m = Matrix(row=2, col=2)
#        m[0] = [1, 0]
#        m[1] = [0, 1]
#        self.assertEquals(m[0], Matrix([1, 0]))
#        self.assertEquals(m[1], Matrix([0, 1]))
#        self.assertEquals(m, Matrix([[1,0], [0,1]]))
#        m = Matrix(row=2, col=2)
#        m[0] = [1, 0, 0]
#        m[1] = [0, 1, 0]
#        self.assertRaises(MismatchingDimensions, m.__setitem__, [1, 0, 0])
#        self.assertRaises(MismatchingDimensions, m.__setitem__, [1, 0, 0])
#        
#
#    def test1DMatrixInit(self):
#        """Matrix should be initialized to a 1D matrix"""
#        m = Matrix([1, 2])
#        self.assertEquals(m.dim, 1)
#        self.assertEquals(m.rows, 1)
#        self.assertEquals(m.cols, 2)
#        self.assertEquals(m[0], 1)
#        self.assertEquals(m[1], 2)
#        m = Matrix([[1],[2]])
#        self.assertEquals(m.dim, 1)
#        self.assertEquals(m.rows, 2)
#        self.assertEquals(m.cols, 1)
#        self.assertEquals(m[0], 1)
#        self.assertEquals(m[1], 2)
#
#    def test2DMatrixInit(self):
#        """Matrix should be initialized to 2D matrix"""
#        m = Matrix([[1, 2], [3, 4], [5, 6]])
#        self.assertEquals(m.dim, 2)
#        self.assertEquals(m.rows, 3)
#        self.assertEquals(m.cols, 2)
#        self.assertEquals(m[0], Matrix([1, 2]))
#        self.assertEquals(m[1], Matrix([3, 4]))
#        self.assertEquals(m[2], Matrix([5, 6]))
#        self.assertEquals(m[0][0], 1)
#        self.assertEquals(m[1][1], 4)
#        self.assertEquals(m[2][0], 5)
#    def testAddMatrix(self):
#        """Matrices of the same dimensions should be added"""
#        m1 = Matrix([1, 2, 3])
#        m2 = Matrix([1, 2, 3])
#        self.assertEquals(m1 + m2, Matrix([2, 4, 6]))
#        m1 = Matrix([[1, 2], [3, 4]])
#        m2 = Matrix([[1, 2], [3, 4]])
#        m3 = m1 + m2
#        self.assertEquals(m1 + m2, Matrix([[2, 4], [6, 8]]))
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
#    def testMismatchingDimensionsAdd(self):
#        """Operations with matrices that don't match should raise exceptions"""
#        m1 = Matrix([1])
#        self.assertRaises(MismatchingDimensions, m1.__add__, Matrix([1, 2]))
#        m1 = Matrix([1, 2])
#        m2 = Matrix([1, 2])
#        self.assertRaises(MismatchingDimensions, m1.__mul__, m2)

if __name__ == "__main__":
    unittest.main()