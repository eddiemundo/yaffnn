import unittest
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

class Matrix:
    def _isNum(self, s):
        """Tests if s is a number"""
        try:
            float(s)
            return True
        except:
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

    def __init__(self, matrix=[]): 
        self.matrix = [] # our internal representation of the matrix
        if matrix == []:
            self.dim = 0
            self.rows = 0
            self.cols = 0
        elif(self._is1d(matrix)):        
            self.dim = 1
            if isinstance(matrix[0], list):
                self.rows = len(matrix)
                self.cols = 1
                for row in matrix:
                    self.matrix += row
            else:
                self.rows = 1
                self.cols = len(matrix)
                self.matrix = matrix[:]
            
        elif(self._is2d(matrix)):
            self.dim = 2
            self.rows = len(matrix)
            self.cols = len(matrix[0])
            for row in matrix:
                self.matrix += row
        else:
            raise MalformedMatrix("Matrix has invalid dimensions or is a block matrix.")


    def __getitem__(self, key):
        if self.dim == 1:
            return self.matrix[key]
        elif self.dim == 2:
            indexStart = self.cols * key
            return Matrix(self.matrix[indexStart:indexStart+self.cols])
        
    def __eq__(self, other):
        return self.matrix == other.matrix and self.rows == other.rows and self.cols == other.cols and self.dim == other.dim
    
    def __add__(self, other):
        if self.rows == other.rows and self.cols == other.cols:
            matrix = []
            for i in range(len(self.matrix)):
                matrix.append(self.matrix[i] + other.matrix[i])
            # hack that uses knowledge of internal representation
            result = Matrix(matrix)
            result.dim = self.dim
            result.rows = self.rows
            result.cols = self.cols
            return result
        else:
            raise MismatchingDimensions("Dimensions of the matrices don't match.")
    
    def __sub__(self, other):
        if self.rows == other.rows and self.cols == other.cols:
            matrix = []
            for i in range(len(self.matrix)):
                matrix.append(self.matrix[i] - other.matrix[i])
            result = Matrix(matrix)
            result.dim = self.dim
            result.rows = self.rows
            result.cols = self.cols
            return result
        else:
            raise MismatchingDimensions("Dimensions of the matrices don't match")
        
    def __mul__(self, other):
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
        result.dim = 2
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

    
class NeuralNetwork:
    """A feedforward neural network"""
    def __init__(self, topology):
        # topology = [[inputs], #hidden nodes, ..., #hidden nodes, #outputs]
        self.inputs = topology[0]
        self.topology = topology
        for i in range(len(topology)):
            #self.weightMatrices[i] =
            None 
        
    topology = [] 
    weightMatrices = [] # array of weight matrices
    hiddenNodes = []
    outputs = []

class MatrixTest(unittest.TestCase):
    def testEmptyInit(self):
        """Matrix should be initialized to empty with no parameters""" 
        m = Matrix()
        self.assertEquals(m.dim, 0)
        self.assertEquals(m.rows, 0)
        self.assertEquals(m.cols, 0)
        self.assertEquals(m.matrix, [])

    def test1DMatrixInit(self):
        """Matrix should be initialized to a 1D matrix"""
        m = Matrix([1, 2])
        self.assertEquals(m.dim, 1)
        self.assertEquals(m.rows, 1)
        self.assertEquals(m.cols, 2)
        self.assertEquals(m[0], 1)
        self.assertEquals(m[1], 2)
        m = Matrix([[1],[2]])
        self.assertEquals(m.dim, 1)
        self.assertEquals(m.rows, 2)
        self.assertEquals(m.cols, 1)
        self.assertEquals(m[0], 1)
        self.assertEquals(m[1], 2)

    def test2DMatrixInit(self):
        """Matrix should be initialized to 2D matrix"""
        m = Matrix([[1, 2], [3, 4], [5, 6]])
        self.assertEquals(m.dim, 2)
        self.assertEquals(m.rows, 3)
        self.assertEquals(m.cols, 2)
        self.assertEquals(m[0], Matrix([1, 2]))
        self.assertEquals(m[1], Matrix([3, 4]))
        self.assertEquals(m[2], Matrix([5, 6]))
        self.assertEquals(m[0][0], 1)
        self.assertEquals(m[1][1], 4)
        self.assertEquals(m[2][0], 5)
    def testAddMatrix(self):
        """Matrices of the same dimensions should be added"""
        m1 = Matrix([1, 2, 3])
        m2 = Matrix([1, 2, 3])
        self.assertEquals(m1 + m2, Matrix([2, 4, 6]))
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[1, 2], [3, 4]])
        m3 = m1 + m2
        self.assertEquals(m1 + m2, Matrix([[2, 4], [6, 8]]))
    def testMulMatrix(self):
        """Matrices of compatible dimensions should be multiplied"""
        m1 = Matrix([[1, 2, 3], [4, 5, 6]])
        m2 = Matrix([[1, 2], [3, 4], [5, 6]])
        m3 = m1*m2
        self.assertEquals(m1 * m2, Matrix([[22, 28], [49, 64]]))
        
    def testTranspose(self):
        m = Matrix([[1, 4], [2, 5], [3, 6]])
        m.transpose()
        self.assertEquals(m, Matrix([[1,2,3],[4,5,6]]))
        m.transpose()
        self.assertEquals(m, Matrix([[1,4],[2,5],[3,6]]))
    def testMalformedMatrixInit(self):
        """Malformed matrices should raise exceptions"""
        self.assertRaises(MalformedMatrix, Matrix, [[1], 2])
        self.assertRaises(MalformedMatrix, Matrix, [[]])
        self.assertRaises(MalformedMatrix, Matrix, [[[1]]])
        self.assertRaises(MalformedMatrix, Matrix, [[1], [2, 3], [4, 5]])

    def testMismatchingDimensionsAdd(self):
        """Operations with matrices that don't match should raise exceptions"""
        m1 = Matrix([1])
        self.assertRaises(MismatchingDimensions, m1.__add__, Matrix([1, 2]))
        m1 = Matrix([1, 2])
        m2 = Matrix([1, 2])
        self.assertRaises(MismatchingDimensions, m1.__mul__, m2)

if __name__ == "__main__":
    unittest.main()