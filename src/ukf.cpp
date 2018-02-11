#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // If this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // If this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // State vector dimension
  n_x_ = 5;

  // Initial state vector
  x_ = VectorXd(n_x_);

  // Initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
  P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.

  // Augmented state dimension
  n_aug_ = n_x_ + 2;
  
  // Sigma point spreading parameter
  lambda_ = 3 - n_x_;

  // Number of sigma points
  n_sig_ = 2 * n_aug_ + 1;

  // Create matrix with predicted sigma points as columns
  Xsig_pred_ = MatrixXd(n_x_, n_sig_);

  // Initialize weights
  weights_ = VectorXd(n_sig_);
  weights_.fill(0.5 / (lambda_ + n_aug_));
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  // Initialize measurement noice covarieance matrices
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << std_radr_ * std_radr_, 0, 0,
              0, std_radphi_ * std_radphi_, 0,
              0, 0, std_radrd_ * std_radrd_;

  R_lidar_ = MatrixXd(2, 2);
  R_lidar_ << std_laspx_ * std_laspx_, 0,
              0, std_laspy_ * std_laspy_;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_) {
    // Initialize state vectors and covariance matrices
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // Extract values for better readibility
      double rho = meas_package.raw_measurements_[0]; // range
      double phi = meas_package.raw_measurements_[1]; // bearing
      double rho_dot = meas_package.raw_measurements_[2]; // velocity of rh
      // Convert from polar to cartesian coordinates
      double px = rho * cos(phi);
      double py = rho * sin(phi);
      double vx = rho_dot * cos(phi);
      double vy = rho_dot * sin(phi);
      double v  = sqrt(vx * vx + vy * vy);
      // Initialize state vector
      x_ << px, py, v, 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // Extract values for better readibility
      double px = meas_package.raw_measurements_[0];
      double py = meas_package.raw_measurements_[1];
      // Initialize state vector
      x_ << px, py, 0, 0, 0;
    }

    // Saving first timestamp in seconds
    time_us_ = meas_package.timestamp_;

    // Initializing done
    is_initialized_ = true;
    // No need to predict or update
    return;
  }

  // Calculate dt
  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0; // in seconds
  time_us_ = meas_package.timestamp_;

  // Prediction step
  Prediction(dt);

  // Update step
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
    UpdateRadar(meas_package);
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
    UpdateLidar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /*******************************************************************************
  *  Sigma Point Augmentation                                                    *
  *******************************************************************************/
  // Create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(n_x_) = x_;
  x_aug(n_x_)   = 0;
  x_aug(n_x_+1) = 0;

  // Create augmented state covariance matrix
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_,n_x_) = P_;
  P_aug(n_x_,n_x_)     = std_a_ * std_a_;
  P_aug(n_x_+1,n_x_+1) = std_yawdd_ * std_yawdd_;

  // Create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  // Create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_);
  Xsig_aug.col(0) = x_aug;
  double lambda_add_n_aug_sqrt = sqrt(lambda_ + n_aug_);
  for (int i=0; i<n_aug_; i++) {
    Xsig_aug.col(i+1)        = x_aug + lambda_add_n_aug_sqrt * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - lambda_add_n_aug_sqrt * L.col(i);
  }

  /*******************************************************************************
  *  Predict Sigma Points                                                        *
  *******************************************************************************/
  // Predict sigma points
  for (int i=0; i<n_sig_; i++) {
    // Extract values for better readability
    double p_x      = Xsig_aug(0,i);
    double p_y      = Xsig_aug(1,i);
    double v        = Xsig_aug(2,i);
    double yaw      = Xsig_aug(3,i);
    double yawd     = Xsig_aug(4,i);
    double nu_a     = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    // Predicted state values
    double px_p, py_p;

    // Avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v/yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v/yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    }
    else {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    // Add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;

    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    // Write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }

  /*******************************************************************************
  *  Predicted Mean and Covariance                                               *
  *******************************************************************************/
  // Predicted state mean vector
  x_ = Xsig_pred_ * weights_;

  // Predicted state covariance matrix
  P_.fill(0.0);
  for (int i=0; i<n_sig_; i++) { // Iterate over sigma points
    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // Angle normalization
    while (x_diff(3) >  M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;
    // Predict state covariance matrix
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  // Set measurement dimension, radar can measure two dimensions
  int n_z = 2;

  /*******************************************************************************
  *  Predict Measurement                                                         *
  *******************************************************************************/
  // Create matrix for sigma points in measurement space
  // Therefore, extract the x and y column from the predicted sigma points matrix
  MatrixXd Zsig = Xsig_pred_.block(0, 0, n_z, n_sig_);

  // Calculate mean predicted measurement
  VectorXd z_pred = Zsig * weights_;

  // Calculate innovation covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for (int i=0; i<n_sig_; i++) { // Iterate over sigma points
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // Add measurement noise covariance matrix
  S = S + R_lidar_;
  
  /*******************************************************************************
  *  Update State                                                                *
  *******************************************************************************/
  // Incoming lidar measurement
  VectorXd z = meas_package.raw_measurements_;

  // Calculate cross correlation matrix Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i=0; i<n_sig_; i++) { // Iterate over sigma points
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Calculate Kalman gain K
  MatrixXd K = Tc * S.inverse();

  // Residual
  VectorXd z_diff = z - z_pred;

  // Update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
  
  // Update lidar NIS
  NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  // Set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  /*******************************************************************************
  *  Predict Measurement                                                         *
  *******************************************************************************/
  // Create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, n_sig_);
  // Transform sigma points into measurement space
  for (int i=0; i<n_sig_; i++) { // Iterate over sigma points
    // Extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v   = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);
    
    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // Measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                      // r
    Zsig(1,i) = atan2(p_y,p_x);                               // phi
    Zsig(2,i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);  // r_dot
  }

  // Calculate mean predicted measurement
  VectorXd z_pred = Zsig * weights_;

  // Calculate innovation covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for (int i=0; i<n_sig_; i++) { // Iterate over sigma points
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    
    // Angle normalization
    while (z_diff(1) >  M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;
    
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // Add measurement noise covariance matrix
  S = S + R_radar_;

  /*******************************************************************************
  *  Update State                                                                *
  *******************************************************************************/
  // Incoming radar measurement
  VectorXd z = meas_package.raw_measurements_;

  // Calculate cross correlation matrix Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i=0; i<n_sig_; i++) { // Iterate over sigma points
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // Angle normalization
    while (z_diff(1) >  M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // Angle normalization
    while (x_diff(3) >  M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Calculate Kalman gain K
  MatrixXd K = Tc * S.inverse();

  // Residual
  VectorXd z_diff = z - z_pred;
  // Angle normalization
  while (z_diff(1) >  M_PI) z_diff(1) -= 2. * M_PI;
  while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

  // Update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  // Update radar NIS
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}
