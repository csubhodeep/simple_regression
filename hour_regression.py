import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error as mse, mean_absolute_error as mae
from sklearn.preprocessing import OneHotEncoder
import logging
import time
from joblib import dump

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        logging.info("Time taken = {}".format(te-ts))
        return result
    return timed


class HourRegression:

    def __init__(self):
        self.mean_absolute_deviations = np.inf

    def dispose(self):
        """
        Deletes this instance from memory
        :return:
        """
        del(self)

    def adjusted_r2_score(self,y_true,y_pred,p):
        """
        This is a custom scoring function that calculates the adjusted R2 score
        :param y_true: The ground true reference
        :param y_pred: The predictions
        :param p: number of parameters or features
        :return: score
        """
        try:
            assert(y_true.shape[0]==y_pred.shape[0])
            n = y_true.shape[0]
            return 1 - (1 - r2_score(y_true, y_pred)) * ((n - 1) / (n - p - 1))
        except AssertionError as error:
            logging.error("Unequal number of observations")

    def get_scores(self,y_true, y_pred,x_test):
        """
        This function calculates some popular metrics for regression by
        comparing the true and predicted values
        :param y_true: the ground true reference
        :param y_pred: the predictions
        :param x_test: this has been kept only to count the total number of features
        :return: None
        """
        try:
            assert(y_true.shape[0]==y_pred.shape[0])
            p = x_test.shape[1] # number of parameters
            print("Adjusted R2 Score {}".format(self.adjusted_r2_score(y_true,y_pred,p)))
            print("RMSE {}".format(np.sqrt(mse(y_true,y_pred))))
            mean_absolute_deviations = mae(y_true,y_pred)
            print("MAE {}".format(mean_absolute_deviations))
            self.mean_absolute_deviations = mean_absolute_deviations
        except AssertionError as error:
            logging.error("Unequal number of observations")


    @timeit
    def get_data(self,path):
        """
        This function reads data from disk
        :param path: A path on the disk
        :return: raw data of type 2d numpy float64 arrays.
        """
        try:
            raw_data = np.genfromtxt(path,delimiter=',',skip_header=1)
            return raw_data
        except:
            logging.error("Invalid path")


    def get_plots(self,y_true,y_pred,x_tick_labels):
        """
        This function plot the predictions against true data points
        :param y_true: true values
        :param y_pred: predictions
        :param x_tick_labels: markers for x-axis
        :return: None
        """
        try:
            assert(y_true.shape[0]==y_pred.shape[0])
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1)
            xc = np.arange(len(x_tick_labels))
            ax1.plot(xc, y_pred, label='pred')
            ax1.plot(xc, y_true, label='true')
            ax1.set_ylabel("Total count of bikes shared hourly")
            plt.legend()
            plt.show()
            return None
        except AssertionError as error:
            logging.error("Unequal number of samples in output")


    def pre_process(self,data):
        """
        this function is responsible to prepare the dataset for training and testing
        :param data: the raw data
        :return: standardised training input and output, scaling object of the output data and a series of instances
        """

        try:
            assert (data.shape[1]==17)
            logging.info("Pre processing raw data")
            # taking the last column - 'cnt' to be the dependent variable
            y = data[:, -1]

            # encode 'season' and 'weather' to multiple columns
            season_cols = OneHotEncoder(sparse=False).fit_transform(data[:,2].reshape(-1,1))
            weather_cols = OneHotEncoder(sparse=False).fit_transform(data[:,9].reshape(-1,1))

            # the second (date) column is redundant and non-numeric therefore has not been considered
            # the first column has been kept only to identify each prediction with its instance and has not been considered in the analysis
            # deleting un-necessary features like registered, casual and total (cnt) counts as they have been considered as dependent variables
            new_data = np.delete(data, [1, 2, 9, 14, 15, 16], 1)
            X = np.hstack((new_data,season_cols,weather_cols))
            # the total data is shuffled and split into training and testing data sets
            x_train, x_test, y_train, y_test = train_test_split(X, y,shuffle=True)

            # initialising the scalers
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()

            # scalers are fit on training data only (throws an error if the input contains non-numeric or NaN or Inf)
            assert(np.any(np.isnan(x_train[:,1:]))!=True)
            scaler_x.fit(x_train[:, 1:])
            scaler_y.fit(y_train.reshape(-1,1))

            # scaling input dataset
            x_train_std = scaler_x.transform(x_train[:, 1:])
            x_test_std = scaler_x.transform(x_test[:, 1:])

            # test instances are saved for latter plotting purposes
            test_instances = x_test[:,0]

            # scaling output dataset
            y_train_std = scaler_y.transform(y_train.reshape(-1, 1))

            return x_train_std, y_train_std, x_test_std, y_test, scaler_y, test_instances
        except AssertionError as error:
            print("input data cannot have NaN or Inf values")


    @timeit
    def train(self,x_train,y_train):
        """
        initiating and fitting an ML model
        :param x_train: training input
        :param y_train: training output
        :return: trained model
        """
        try:
            assert (x_train.shape[0]==y_train.shape[0])
            logging.info("Training model")
            #model = MLPRegressor(hidden_layer_sizes=[100, 100])
            #model = RandomForestRegressor()
            model = DecisionTreeRegressor()
            model.fit(x_train, y_train)
            return model
        except AssertionError as error:
            logging.error("Unequal number of samples")

    def post_process(self,y_pred,scaler):
        """
        inverting the predictions to their original scale and rounding them to the nearest integer
        :param y_pred: raw predictions
        :param scaler: scaling obejct for the output data
        :return: transformed predictions
        """
        logging.info("Post processing predictions")
        # the predictions have been rounded to the nearest integer as originally the 'cnt' column was integer
        y_processed = np.round(scaler.inverse_transform(y_pred.reshape(-1,1)))
        return y_processed

    def predict(self,model,input):
        """
        This function gives predictions for a certain input
        :param model: trained model
        :param input: (test) input
        :return: predictions
        """
        logging.info("Predicting")
        output = model.predict(input)
        return output

    def main(self,path):
        """
        this is the main function that initiates the pipeline
        :param path: the path on the disk where the data is kept
        :return: None
        """
        try:
            assert(path!="")
            logging.info("Starting pipeline")
            data = self.get_data(path)

            # checking if the data has been imported correctly
            logging.info("Shape of data imported: "+str(data.shape))

            # pre processing the data
            x_train, y_train, x_test, y_test, scaler_y, test_instances = self.pre_process(data)

            # training
            self.model = self.train(x_train,y_train)


            # making predictions on the transformed dataset
            y_pred_raw = self.predict(self.model,x_test)

            # inverting the predictions to their original scale
            y_pred = self.post_process(y_pred_raw,scaler_y)

            # generating scores
            self.get_scores(y_test,y_pred,x_test)

            # persist model if model-accuracy is satisfactory
            if self.mean_absolute_deviations < 1:
                dump(self.model,"model.pkl")

            # generating plots
            self.get_plots(y_test,y_pred,test_instances)

            return None
        except AssertionError as error:
            logging.error("Path cannot be null")



if __name__ == "__main__":
    path = "data/hour.csv"
    HourRegression().main(path)




