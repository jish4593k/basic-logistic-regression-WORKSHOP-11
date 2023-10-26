#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <mlpack/core.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>

using namespace std;
using namespace cv;
using namespace mlpack::regression;

int main() {
    // Load the dataset
    mlpack::data::Load("  v", X, true);
    arma::Row<size_t> y = X.row(X.n_rows - 1);
    X.shed_row(X.n_rows - 1);

    // Split the dataset into training and test sets
    arma::mat X_train, X_test;
    arma::Row<size_t> y_train, y_test;
    mlpack::data::Split(X, y, X_train, X_test, y_train, y_test, 0.25);

    // Feature scaling
    mlpack::preprocessing::StandardScaler scaler;
    scaler.Fit(X_train);
    X_train = scaler.Transform(X_train);
    X_test = scaler.Transform(X_test);

    // Fit the logistic regression model
    LogisticRegression lr(X_train, y_train, 0.0);

    // Predict the test set results
    arma::Row<size_t> y_pred;
    lr.Predict(X_test, y_pred);

    // Evaluate the model
    mlpack::regression::L2LogisticRegression lr2(X_train, y_train, 0.0);
    arma::Row<size_t> y_pred_prob;
    lr2.Predict(X_test, y_pred_prob);

    arma::Mat<size_t> confusion;
    mlpack::regression::LogisticRegression::CalculateConfusion(y_test, y_pred, confusion);

    double accuracy = (confusion(0, 0) + confusion(1, 1)) / (double)y_test.n_elem;

    cout << "Confusion Matrix:\n" << confusion << endl;
    cout << "Accuracy: " << accuracy << endl;

    // Visualize the training set results (assuming 2D data)
    arma::mat X_set, Y_set;
    X_set = X_train;
    Y_set = y_train;

    Mat image = Mat::zeros(600, 600, CV_8UC3);

    for (int i = 0; i < X_set.n_cols; ++i) {
        int x = (X_set(0, i) - X_set.row(0).min()) / (X_set.row(0).max() - X_set.row(0).min()) * 600;
        int y = (X_set(1, i) - X_set.row(1).min()) / (X_set.row(1).max() - X_set.row(1).min()) * 600;

        if (Y_set(0, i) == 0) {
            circle(image, Point(x, y), 5, Scalar(0, 0, 255), -1);
        }
        else {
            circle(image, Point(x, y), 5, Scalar(0, 255, 0), -1);
        }
    }

    imshow("Training Set", image);
    waitKey(0);

    return 0;
}
