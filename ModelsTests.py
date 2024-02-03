from RegressionModels import perform_svm
import random
import numpy as np
import matplotlib.pyplot as plt

# Linear Relationship 1D TestData
X_train_linear_1D = np.random.rand(90)
y_train_linear_1D = [(3*x+1)*(0.95+0.1*random.random()) for x in X_train_linear_1D]
X_test_linear_1D = np.random.rand(10)
y_test_linear_1D = [(3*x+1)*(0.95+0.1*random.random()) for x in X_test_linear_1D]

# Linear Relationship 2D Test Data (Planar Data)
X_train_linear_2D = np.random.rand(90, 2)
y_train_linear_2D = [(3*x[0]+4*x[1]+1)*(0.95+0.1*random.random()) for x in X_train_linear_2D]
X_test_linear_2D = np.random.rand(10, 2)
y_test_linear_2D = [(3*x[0]+4*x[1]+1)*(0.95+0.1*random.random()) for x in X_test_linear_2D]

# Linear Relationship 5D Test Data (Hyper Planar Data)
X_train_linear_5D = np.random.rand(450, 5)
y_train_linear_5D = [(3*x[0] + 4*x[1] + 2*x[3] +4)*(0.95+0.1*random.random()) for x in X_train_linear_5D]
X_test_linear_5D = np.random.rand(50, 5)
y_test_linear_5D = [(3*x[0] + 4*x[1] + 2*x[3] +4)*(0.95+0.1*random.random()) for x in X_test_linear_5D]

# NonLinear Relationship 1D TestData
X_train_nonlinear_1D = np.random.rand(90, 1)
y_train_nonlinear_1D = [(3 * x ** 2 + x + 1)*(0.95+0.1*random.random()) for x in X_train_nonlinear_1D]
X_test_nonlinear_1D = np.random.rand(10, 1)
y_test_nonlinear_1D = [(3 * x ** 2 + x + 1)*(0.95+0.1*random.random()) for x in X_test_nonlinear_1D]

# NonLinear Relationship 2D Test Data (Planar Data)
X_train_nonlinear_2D = np.random.rand(90, 2)
y_train_nonlinear_2D = [(3*x[0]+4*x[1]-4*x[1]**2+1)*(0.95+0.1*random.random()) for x in X_train_nonlinear_2D]
X_test_nonlinear_2D = np.random.rand(10, 2)
y_test_nonlinear_2D = [(3*x[0]+4*x[1]-4*x[1]**2+1)*(0.95+0.1*random.random()) for x in X_test_nonlinear_2D]

# NonLinear Relationship 5D Test Data
X_train_nonlinear_5D = np.random.rand(450, 5)
y_train_nonlinear_5D = [(3*x[0]**2 + 4*x[1] - 2*x[0] + 2*x[3]**3 + 4)*(0.95+0.1*random.random()) for x in X_train_nonlinear_5D]
X_test_nonlinear_5D = np.random.rand(50, 5)
y_test_nonlinear_5D = [(3*x[0]**2 + 4*x[1] - 2*x[0] + 2*x[3]**3 + 4)*(0.95+0.1*random.random()) for x in X_test_nonlinear_5D]


fig, axs = plt.subplots(2, 2)  # Create a figure containing 4 axes subplots.

# plot the 1D and 2D data
axs[0,0].plot(X_train_linear_1D, y_train_linear_1D, 'ro', X_test_linear_1D, y_test_linear_1D, 'bo')
axs[0,0].set_title('Linear Test Data 1D')
axs[1,0].scatter(X_train_linear_2D[:,0], X_train_linear_2D[:,1], c = y_train_linear_2D)
axs[1,0].set_title('Linear Test Data 2D')
axs[1,0].scatter(X_test_linear_2D[:,0], X_test_linear_2D[:,1], c = y_test_linear_2D, marker = 's')
axs[1,0].set_title('Linear Test Data 2D')
axs[0,1].plot(X_train_nonlinear_1D, y_train_nonlinear_1D, 'ro', X_test_nonlinear_1D, y_test_nonlinear_1D, 'bo')
axs[0,1].set_title('Linear Test Data 1D')
axs[1,1].scatter(X_train_nonlinear_2D[:,0], X_train_nonlinear_2D[:,1], c = y_train_linear_2D)
axs[1,1].set_title('Linear Test Data 2D')
axs[1,1].scatter(X_test_nonlinear_2D[:,0], X_test_nonlinear_2D[:,1], c = y_test_linear_2D, marker = 's')
axs[1,1].set_title('Linear Test Data 2D')

# print(X_train_linear_1D)
# print(X_train_linear_1D.reshape(-1, 1))


#plot the 1D case summary
linear_1D_summary = perform_svm(X_train_linear_1D.reshape(-1, 1),
                                y_train_linear_1D, X_test_linear_1D.reshape(-1, 1), y_test_linear_1D, max_iter=1000)
evaluation_figure_1D, eval_axs_1D = plt.subplots(3, 1)
plt.xticks(rotation=90)


eval_axs_1D[0].bar(linear_1D_summary[0].keys(), [i[0] for i in linear_1D_summary[0].values()])
eval_axs_1D[1].bar(linear_1D_summary[0].keys(), [i[1] for i in linear_1D_summary[0].values()])
eval_axs_1D[2].bar(linear_1D_summary[0].keys(), [i[2] for i in linear_1D_summary[0].values()])
eval_axs_1D[0].bar(linear_1D_summary[1].keys(), [i[0] for i in linear_1D_summary[1].values()])
eval_axs_1D[1].bar(linear_1D_summary[1].keys(), [i[1] for i in linear_1D_summary[1].values()])
eval_axs_1D[2].bar(linear_1D_summary[1].keys(), [i[2] for i in linear_1D_summary[1].values()])



# nonlinear_1D_summary = perform_svm(X_train_nonlinear_1D.reshape(-1, 1),
#                                    y_train_nonlinear_1D,
#                                    X_test_nonlinear_1D.reshape(-1, 1),
#                                    y_test_nonlinear_1D, max_iter=100)
# evaluation_figure_nonlinear_1D, eval_axs_nonlinear_1D = plt.subplots(3, 1)
# eval_axs_nonlinear_1D[0].plot(nonlinear_1D_summary[0], nonlinear_1D_summary[1], 'b-')
# eval_axs_nonlinear_1D[1].plot(nonlinear_1D_summary[0], nonlinear_1D_summary[2], 'b-')
# eval_axs_nonlinear_1D[2].plot(nonlinear_1D_summary[0], nonlinear_1D_summary[3], 'b-')
# eval_axs_nonlinear_1D[0].plot(nonlinear_1D_summary[0], nonlinear_1D_summary[4], 'r-')
# eval_axs_nonlinear_1D[1].plot(nonlinear_1D_summary[0], nonlinear_1D_summary[5], 'r-')
# eval_axs_nonlinear_1D[2].plot(nonlinear_1D_summary[0], nonlinear_1D_summary[6], 'r-')



# linear_2D_summary = perform_svm(X_train_linear_2D,
#                                 y_train_linear_2D, X_test_linear_2D, y_test_linear_2D, max_iter=100)
# print(linear_2D_summary)
# print(linear_2D_summary[0])
# evaluation_figure_2D, eval_axs_2D = plt.subplots(3, 1)
# eval_axs_2D[0].plot(linear_2D_summary[0], linear_2D_summary[1], 'b-')
# eval_axs_2D[1].plot(linear_2D_summary[0], linear_2D_summary[2], 'b-')
# eval_axs_2D[2].plot(linear_2D_summary[0], linear_2D_summary[3], 'b-')
# eval_axs_2D[0].plot(linear_2D_summary[0], linear_2D_summary[4], 'r-')
# eval_axs_2D[1].plot(linear_2D_summary[0], linear_2D_summary[5], 'r-')
# eval_axs_2D[2].plot(linear_2D_summary[0], linear_2D_summary[6], 'r-')
#
#
#
# nonlinear_2D_summary = perform_svm(X_train_nonlinear_2D,
#                                    y_train_nonlinear_2D,
#                                    X_test_nonlinear_2D,
#                                    y_test_nonlinear_2D, max_iter=100)
# evaluation_figure_nonlinear_2D, eval_axs_nonlinear_2D = plt.subplots(3, 1)
# eval_axs_nonlinear_2D[0].bar(nonlinear_2D_summary[0], nonlinear_2D_summary[1])
# eval_axs_nonlinear_2D[1].bar(nonlinear_2D_summary[0], nonlinear_2D_summary[2])
# eval_axs_nonlinear_2D[2].bar(nonlinear_2D_summary[0], nonlinear_2D_summary[3])
# eval_axs_nonlinear_2D[0].bar(nonlinear_2D_summary[0], nonlinear_2D_summary[4])
# eval_axs_nonlinear_2D[1].bar(nonlinear_2D_summary[0], nonlinear_2D_summary[5])
# eval_axs_nonlinear_2D[2].bar(nonlinear_2D_summary[0], nonlinear_2D_summary[6])

plt.show()
