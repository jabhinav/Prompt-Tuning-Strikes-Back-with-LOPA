from matplotlib import pyplot as plt


# # For Selector Net
# X-axis: Prior Net Training Iter, Y-axis: Test Data pass@1
# y_axis = [16.42, 16.42, 16.83, 16.83, 16.63, 16.63, 16.42, 16.63, 17.24, 17.04, 17.24, 17.24]  # 0.50: 0.50
x_axis = [0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
y_axis = [11.90, 11.70, 11.49, 11.49, 12.11, 11.29, 9.85, 10.67, 9.44, 9.65, 9.44, 9.65]  # 0.25: 0.75

# Pass@1 for Others
baseline = 11.00
oracle = 18.89
clf_0 = 10.06
clf_1 = 11.29
clf_2 = 9.65
clf_3 = 10.88
clf_4 = 11.90

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