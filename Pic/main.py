import numpy as np
import matplotlib.pyplot as plt

# Generate x values
x = np.linspace(0, 8, 9)  # Values from 0 to 8 (inclusive)

# Generate y values (exponential growth)
y = 2 ** x  # Exponential function

# Plot the graph
plt.figure(figsize=(8, 6))
plt.plot(x, y, marker='o', linestyle='-', color='b')

# Add title and labels
plt.title("Exponential Growth")
plt.xlabel("x")
plt.ylabel("f(x)")

# Add grid
plt.grid()

# Show the plot
plt.show()