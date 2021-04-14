import pandas as pandas
import numpy as np
import matplotlib.pyplot as plt
import sys

# Parameters for normalization
max_y = 0
min_y = 0

max_x = 0
min_x = 0

# Parameters for gradient descent
theta0 = 0
theta1 = 0


def load_train_data(path):
    df = pandas.read_csv(path)
    return df


def plot_data(dataframe):
    dataframe.plot.scatter(x="Width", y="Weight")
    plt.show()


def boxplot(df):
    df.boxplot(column=["Weight", "Width"])
    plt.show()


def plot_lin_reg_func(dfr, y_pred):
    plt.scatter(dfr['Width'], dfr['Weight'])
    plt.plot([min(dfr['Width']), max(dfr['Width'])], [min(y_pred), max(y_pred)], color='red')  # regression line
    plt.show()


def remove_outliers(df):
    for index, row in df.iterrows():
        x, y = row['Weight'], row['Width']
        if (x < 200 and y > 5) or x > 1500 or x == 0:
            df = df.drop(labels=index, axis=0)

    return df


def normalization_train(df):
    # Weight(x)
    global max_x, min_x
    max_x = np.max(df['Weight'])
    min_x = np.min(df['Weight'])
    df['Weight'] = np.array([(x - min_x) / (max_x - min_x) for x in df['Weight']])

    # Width(y)
    global max_y, min_y
    max_y = np.max(df['Width'])
    min_y = np.min(df['Width'])
    df['Width'] = np.array([(x - min_y) / (max_y - min_y) for x in df['Width']])

    return df

def normalization_test(df):
    # Weight(x)
    global max_x, min_x
    df['Weight'] = np.array([(x - min_x) / (max_x - min_x) for x in df['Weight']])

    # Width(y)
    global max_y, min_y
    df['Width'] = np.array([(x - min_y) / (max_y - min_y) for x in df['Width']])

    return df


def denormalization(y_pred):
    global max_x, min_x
    return np.array([x*(max_x - min_x) + min_x for x in y_pred])


def fit(x, y, l):
    # Gradient descent algorithm (l - learning rate)

    global theta0, theta1

    epochs = 400  # The number of iterations to perform gradient descent

    n = float(len(x))  # Number of elements in x

    # Performing Gradient Descent
    for i in range(epochs):
        y_pred = theta1 * x + theta0  # The current predicted value of Y
        d_theta1 = (-2 / n) * sum(x * (y - y_pred))  # Derivative wrt teta1
        d_theta0 = (-2 / n) * sum(y - y_pred)  # Derivative wrt teta0
        theta1 = theta1 - l * d_theta1  # Update teta1
        theta0 = theta0 - l * d_theta0  # Update teta0


def normal_equation(x, y):
    n = len(x)
    x_bias = np.ones((n, 1))
    x = np.reshape(x.values, (n, 1))
    x = np.append(x_bias, x, axis=1)
    x_transpose = np.transpose(x)
    x_transpose_dot_x = x_transpose.dot(x)
    temp_1 = np.linalg.inv(x_transpose_dot_x)
    temp_2 = x_transpose.dot(y)
    theta = temp_1.dot(temp_2)
    global theta0, theta1
    theta0 = theta[0]
    theta1 = theta[1]


def predict(x):
    y_pred = theta1 * x + theta0
    return y_pred


def calculate_rmse(y_true, y_predict):

    rmse = np.sqrt(np.square(np.subtract(y_true, y_predict)).mean())

    return rmse


if __name__ == '__main__':

    train_path = sys.argv[1]
    test_path = sys.argv[2]

    dfr_train = load_train_data(train_path)

    #plot original dataset
    # plot_data(dfr_train)
    # boxplot(dfr_train)

    # plot dataset after removing outliers
    dfr_no_out = remove_outliers(dfr_train)
    #dfr = new_dfr.copy()
    # plot_data(dfr_no_out)
    # boxplot(dfr_no_out)

    # box plot and plot data after normalization
    dfr_train_norm = normalization_train(dfr_no_out)
    # boxplot(dfr_train_norm)

    # Gradient descent
    # fit(dfr_train_norm['Width'], dfr_train_norm['Weight'], 0.1)

    #Normal equation
    normal_equation(dfr_train_norm['Width'], dfr_train_norm['Weight'])
    #train data
    # Y_pred = predict(dfr_train_norm['Width'])

    #plot linear regression function
    # plot_lin_reg_func(dfr_train_norm, Y_pred)

    #Y_pred_denorm = denormalization(Y_pred)

    #print(str(calculate_rmse(dfr['Weight'], Y_pred_denorm)))

    #test data
    dfr_test = load_train_data(test_path)

    dfr_test_copy = dfr_test.copy()

    #plot_data(dfr_test)

    dfr_test_norm = normalization_test(dfr_test_copy)

    Y_pred_test = predict(dfr_test_norm['Width'])

    #plot linear regression function
    # plot_lin_reg_func(dfr_test_norm, Y_pred_test)

    Y_pred_denorm_test = denormalization(Y_pred_test)

    # plt.scatter(dfr_test['Width'], Y_pred_denorm_test)
    # plt.show()

    print(str(calculate_rmse(dfr_test['Weight'], Y_pred_denorm_test)))