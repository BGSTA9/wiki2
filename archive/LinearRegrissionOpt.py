import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Random Sample Data
np.random.seed(0)
x = np.random.rand(100, 1) * 10
y = 2.5 * x.flatten() + np.random.rand(100) * 5

# Creating the linear regression model
model = LinearRegression()

# Creating a figure and four axes
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
(ax1, ax2), (ax3, ax4) = axs

# Initializing lists to store slopes, R-squared values, and MSEs
slopes = []
r2_values = []
mse_values = []

def animate(i):
    # Training the model using the first i+1 data points
    model.fit(x[:i+1], y[:i+1])
    
    # Making predictions using the trained model
    y_pred = model.predict(x)
    
    # Calculating the slope, intercept, R-squared, and MSE
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    # Storing the slope, R-squared, and MSE
    slopes.append(slope)
    r2_values.append(r2)
    mse_values.append(mse)
    
    # Clearing previous plots, otherwise previous lines will be visible on the plot
    for ax in axs.flat:
        ax.cla()
    
    # Plotting the data points and the linear regression line
    ax1.scatter(x, y, color='blue', alpha=0.5)
    ax1.plot(x, y_pred, color='red')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title("Linear Regression")
    
    # Display of the linear equation
    equation_text = f'y = {slope:.2f}x + {intercept:.2f}'
    ax1.text(0.05, 0.95, equation_text, transform=ax1.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))
    
    # Plotting the rate of change (slope) over time, with respect to the model
    ax2.plot(range(1, len(slopes) + 1), slopes, color='green')
    ax2.set_xlabel('Number of Data Points')
    ax2.set_ylabel('Slope')
    ax2.set_title("Rate of Change of Slope")
    
    # Plotting R-squared over time
    ax3.plot(range(1, len(r2_values) + 1), r2_values, color='purple')
    ax3.set_xlabel('Number of Data Points')
    ax3.set_ylabel('R-squared')
    ax3.set_title("R-squared Over Time")
    
    # Plotting MSE over time
    ax4.plot(range(1, len(mse_values) + 1), mse_values, color='orange')
    ax4.set_xlabel('Number of Data Points')
    ax4.set_ylabel('MSE')
    ax4.set_title("MSE Over Time")

    # Plotting residuals (red dots)
    residuals = y - y_pred
    ax1.scatter(x, residuals, color='red', alpha=0.5)
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.set_title("Residuals")

# Creating the animation
ani = FuncAnimation(fig, animate, frames=len(x), repeat=False)

# Saving the animation as a video file
ani.save('Linear_Regression_Metrics.mp4', writer='ffmpeg')

plt.show()
