import tensorflow as tf

x=tf.constant([[3,5,7], #Crear una matriz de tensores
              [4,6,8],
              ])

y=x[:,2] #Seleccionar la columna 2 de la matriz x

print(y)

#x<2
f=tf.Variable(2.0 , dtype=tf.float32 , name="variable") #Crear una variable de tipo flotante

#Asignarle un valor
f.assign(3.0)
#sumarle 2
f.assign_add(2.0)
#restarle 2
f.assign_sub(1.0)

#Mutiplicar 2 matrices constante y variable
h=tf.constant([[1,2,3]])
l=tf.Variable([[4,5,6]])
tf.matmul(h,l) #

#El gradiente almacena toda la información de la derivada parcial de una función multivariableEl gradiente
# almacena toda la información de la derivada parcial de una función multivariable
def documentation():
    c = tf.constant([[4.0, 5.0], [10.0, 1.0]])

    rank_2_tensor = tf.constant([[1, 2],
                                 [3, 4],
                                 [5, 6]], dtype=tf.float32)

#Tensors have shapes. Some vocabulary:

#Shape: The length (number of elements) of each of the dimensions of a tensor.
#Rank: Number of tensor dimensions. A scalar has rank 0, a vector has rank 1, a matrix is rank 2.
#Axis or Dimension: A particular dimension of a tensor.
#Size: The total number of items in the tensor, the product shape vector
#Note: Although you may see reference to a "tensor of two dimensions", a rank-2 tensor does not usually describe a 2D space.

#Tensors and tf.TensorShape objects have convenient properties for accessing these:

# Find the largest value
 print(tf.reduce_max(c))
#
# Find the index of the largest value
 print(tf.argmax(c))
# Compute the softmax
 print(tf.nn.softmax(c))

a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[1, 1],
                 [1, 1]]) # Could have also said `tf.ones([2,2])`

print(tf.add(a, b), "\n")
print(tf.multiply(a, b), "\n")
print(tf.matmul(a, b), "\n")


rank_4_tensor = tf.zeros([3, 2, 4, 5])
print("Type of every element:", rank_4_tensor.dtype)
print("Number of dimensions:", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy())

# Get row and column tensors
print("Second row:", rank_2_tensor[1, :].numpy())
print("Second column:", rank_2_tensor[:, 1].numpy())
print("Last row:", rank_2_tensor[-1, :].numpy())
print("First item in last column:", rank_2_tensor[0, -1].numpy())
print("Skip the first row:")
print(rank_2_tensor[1:, :].numpy(), "\n")

# Shape returns a `TensorShape` object that shows the size on each dimension
var_x = tf.Variable(tf.constant([[1], [2], [3]]))
print(var_x.shape)




