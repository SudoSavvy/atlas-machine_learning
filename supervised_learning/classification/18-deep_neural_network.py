# Let's assume you want a network with 5 input features and 3 layers.
nx = 5  # Number of input features
sizes = [5, 4, 3, 1]  # Number of neurons in each layer

# Initialize the network
d = DeepNeuralNetwork(nx, sizes)

# Create some sample input data X with shape (nx, m) where m is the number of examples
X = np.random.randn(nx, 10)

# Perform forward propagation
A, cache = d.forward_prop(X)

# Print the output
print("Output of the network:")
print(A)
