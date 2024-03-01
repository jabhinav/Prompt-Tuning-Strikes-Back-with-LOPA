from matplotlib import pyplot as plt


# # For Selector Net
# X-axis: Prior Net Training Iter, Y-axis: Test Data pass@1
# y_axis = [16.42, 16.42, 16.83, 16.83, 16.63, 16.63, 16.42, 16.63, 17.24, 17.04, 17.24, 17.24]  # 0.50: 0.50
# y_axis = [11.90, 11.70, 11.49, 11.49, 12.11, 11.29, 9.85, 10.67, 9.44, 9.65, 9.44, 9.65]  # 0.25: 0.75
# y_axis = [17.24, 17.04, 15.81, 17.45, 16.42, 17.04, 17.45, 17.24, 16.42, 16.22, 16.42, 16.01]  # 0.75: 0.25
y_axis = [16.42, 16.42, 16.83, 16.22, 16.63, 14.98, 16.83, 16.63, 17.04, 17.04, 16.22, 16.63]  # 0.75: 0.25

x_axis = [0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# Pass@1 for Others
baseline = 15.72
oracle = 24.64
clf_0 = 16.42
clf_1 = 16.42
clf_2 = 17.24
clf_3 = 17.04
clf_4 = 16.42

plt.xlabel("Training (on split 2) Epochs")
plt.ylabel("Test Data pass@1")

# Now plot for selector net with dashed line
plt.plot(x_axis, y_axis, color='b', linestyle='-', label='Selector Net')

# All dash types = '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'

# Plot for others as horizontal lines with different colors
plt.axhline(y=baseline, color='k', linestyle='--', label='Baseline')
plt.axhline(y=oracle, color='r', linestyle=':', label='Oracle')
plt.axhline(y=clf_0, color='c', linestyle='-', label='CLF 0')
plt.axhline(y=clf_1, color='m', linestyle='-', label='CLF 1')
plt.axhline(y=clf_2, color='y', linestyle='-', label='CLF 2')
plt.axhline(y=clf_3, color='g', linestyle='-', label='CLF 3')
plt.axhline(y=clf_4, color='grey', linestyle='-', label='CLF 4')

plt.legend()

# Save the plot as high quality plot
plt.savefig('./performance.png', dpi=600)