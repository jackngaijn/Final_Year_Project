import numpy as np
import matplotlib.pyplot as plt

# Generate x values
x = np.linspace(0, 15, 9)  # Values from 0 to 8 (inclusive)

# Generate y values (exponential growth)
y1 = 0.5 ** x  # Exponential function
y2 = 2 ** x  # Exponential function

# Plot the graph
fig = plt.figure()
plt.figure(figsize=(8,4))

plt.subplot(211)
plt.plot(x, y1, marker='o', linestyle='-', color='b')
plt.title("Exponential Grow")
plt.xlabel("x")
plt.ylabel("f(x)")

plt.subplot(212)
plt.plot(x, y2, marker='o', linestyle='-', color='r')
plt.xlabel("x")
plt.ylabel("f(x)")

# Save the figure as a PNG file
plt.savefig("exponential_growth.png", dpi=300)  # Save with high resolution (300 DPI)

# Show the plot
plt.show()