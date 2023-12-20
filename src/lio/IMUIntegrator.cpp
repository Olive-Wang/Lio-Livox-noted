#include "IMUIntegrator/IMUIntegrator.h"

IMUIntegrator::IMUIntegrator(){
  Reset();
  noise.setZero();
  noise.block<3, 3>(0, 0) =  Eigen::Matrix3d::Identity() * gyr_n * gyr_n;
  noise.block<3, 3>(3, 3) =  Eigen::Matrix3d::Identity() * acc_n * acc_n;
  noise.block<3, 3>(6, 6) =  Eigen::Matrix3d::Identity() * gyr_w * gyr_w;
  noise.block<3, 3>(9, 9) =  Eigen::Matrix3d::Identity() * acc_w * acc_w;
}

/** \brief constructor of IMUIntegrator
 * \param[in] vIMU: IMU messages need to be integrated
 */
IMUIntegrator::IMUIntegrator(std::vector<sensor_msgs::ImuConstPtr> vIMU):
vimuMsg(std::move(vIMU)){
  Reset();
  noise.setZero();
  noise.block<3, 3>(0, 0) =  Eigen::Matrix3d::Identity() * gyr_n * gyr_n;
  noise.block<3, 3>(3, 3) =  Eigen::Matrix3d::Identity() * acc_n * acc_n;
  noise.block<3, 3>(6, 6) =  Eigen::Matrix3d::Identity() * gyr_w * gyr_w;
  noise.block<3, 3>(9, 9) =  Eigen::Matrix3d::Identity() * acc_w * acc_w;
}

void IMUIntegrator::Reset(){
  dq.setIdentity();
  dp.setZero();
  dv.setZero();
  dtime = 0;
  covariance.setZero();
  jacobian.setIdentity();
  linearized_bg.setZero();
  linearized_ba.setZero();
}

const Eigen::Quaterniond & IMUIntegrator::GetDeltaQ() const {return dq;}

const Eigen::Vector3d & IMUIntegrator::GetDeltaP() const {return dp;}

const Eigen::Vector3d & IMUIntegrator::GetDeltaV() const {return dv;}

const double & IMUIntegrator::GetDeltaTime() const {return dtime;}

const Eigen::Vector3d & IMUIntegrator::GetBiasGyr() const {return linearized_bg;}

const Eigen::Vector3d& IMUIntegrator::GetBiasAcc() const {return linearized_ba;}

const Eigen::Matrix<double, 15, 15>& IMUIntegrator::GetCovariance(){return covariance;}

const Eigen::Matrix<double, 15, 15> & IMUIntegrator::GetJacobian() const {return jacobian;}

void IMUIntegrator::PushIMUMsg(const sensor_msgs::ImuConstPtr& imu){
  vimuMsg.push_back(imu);
}
void IMUIntegrator::PushIMUMsg(const std::vector<sensor_msgs::ImuConstPtr>& vimu){
  vimuMsg.insert(vimuMsg.end(), vimu.begin(), vimu.end());
}
const std::vector<sensor_msgs::ImuConstPtr> & IMUIntegrator::GetIMUMsg() const {return vimuMsg;}

/**
 * 李群	                SO(3)	                        SE(3)
  旋转矩阵构建	        Sophus::SO3d SO3(R)	          Sophus::SE3d SE3(R,t)
  四元数构建	          Sophus::SO3d SO3(q)	          Sophus::SO3d SO3(q,t)
  输出	               SO3.matrix()	                SE3.matrix()
  对数映射	            Vector3d so3=SO3.log()	     Vecotr6d se3=SE3.log()
  指数映射	            SO3d::exp(so3)	             SE3d::exp(se3)
  向量到反对称矩阵	     SO3d::hat(so3)	              SE3d::hat(se3)
  反对称矩阵到向量	     SO3d::vee(hat)	              SE3d::vee(hat)

*/
/**
  * @brief: IMU数据积分
  * @param lastTime  当前帧雷达的时间
  * @return {void}
  */	
void IMUIntegrator::GyroIntegration(double lastTime){
  double current_time = lastTime;
  for(auto & imu : vimuMsg){
    Eigen::Vector3d gyr;
    gyr << imu->angular_velocity.x,
            imu->angular_velocity.y,
            imu->angular_velocity.z;
    double dt = imu->header.stamp.toSec() - current_time;
    ROS_ASSERT(dt >= 0);
    Eigen::Matrix3d dR = Sophus::SO3d::exp(gyr*dt).matrix();  
    Eigen::Quaterniond qr(dq*dR);
    if (qr.w()<0)
      qr.coeffs() *= -1;
    dq = qr.normalized();
    current_time = imu->header.stamp.toSec();
  }
}

/**
  * @brief: IMU预积分，函数中进行IMU的预积分，状态直接累加的，然后维护每帧的雅克比和协方差
  * @param lastTime  当前帧雷达的时间
  * @param ba_ 加速度bias
  * @param bg_ 角速度bias
  * @return {void}
  */	
void IMUIntegrator::PreIntegration(double lastTime, const Eigen::Vector3d& bg, const Eigen::Vector3d& ba){
  Reset();
  linearized_bg = bg;
  linearized_ba = ba;
  double current_time = lastTime;
  for(auto & imu : vimuMsg){
    Eigen::Vector3d gyr;
    gyr <<  imu->angular_velocity.x,
            imu->angular_velocity.y,
            imu->angular_velocity.z;
    Eigen::Vector3d acc;
    acc << imu->linear_acceleration.x * gnorm,
            imu->linear_acceleration.y * gnorm,
            imu->linear_acceleration.z * gnorm;
    double dt = imu->header.stamp.toSec() - current_time;
    if(dt <= 0 )
      ROS_WARN("dt <= 0");
    gyr -= bg;
    acc -= ba;
    double dt2 = dt*dt;
    Eigen::Vector3d gyr_dt = gyr*dt;
    Eigen::Matrix3d dR = Sophus::SO3d::exp(gyr_dt).matrix();
    Eigen::Matrix3d Jr = Eigen::Matrix3d::Identity();
    double gyr_dt_norm = gyr_dt.norm();
    if(gyr_dt_norm > 0.00001){
      Eigen::Vector3d k = gyr_dt.normalized();
      //使用SO3d::hat()可以实现so3->SO3的转换。
      Eigen::Matrix3d K = Sophus::SO3d::hat(k);
      Jr =   Eigen::Matrix3d::Identity()
             - (1-cos(gyr_dt_norm))/gyr_dt_norm*K
             + (1-sin(gyr_dt_norm)/gyr_dt_norm)*K*K;
    }
    //计算A矩阵和B矩阵， 更新雅克比矩阵和协方差矩阵
    Eigen::Matrix<double,15,15> A = Eigen::Matrix<double,15,15>::Identity();
    A.block<3,3>(0,3) = -0.5*dq.matrix()*Sophus::SO3d::hat(acc)*dt2;
    A.block<3,3>(0,6) = Eigen::Matrix3d::Identity()*dt;
    A.block<3,3>(0,12) = -0.5*dq.matrix()*dt2;
    A.block<3,3>(3,3) = dR.transpose();
    A.block<3,3>(3,9) = - Jr*dt;
    A.block<3,3>(6,3) = -dq.matrix()*Sophus::SO3d::hat(acc)*dt;
    A.block<3,3>(6,12) = -dq.matrix()*dt;
    Eigen::Matrix<double,15,12> B = Eigen::Matrix<double,15,12>::Zero();
    B.block<3,3>(0,3) = 0.5*dq.matrix()*dt2;
    B.block<3,3>(3,0) = Jr*dt;
    B.block<3,3>(6,3) = dq.matrix()*dt;
    B.block<3,3>(9,6) = Eigen::Matrix3d::Identity()*dt;
    B.block<3,3>(12,9) = Eigen::Matrix3d::Identity()*dt;
    jacobian = A * jacobian;
    covariance = A * covariance * A.transpose() + B * noise * B.transpose();
    
    // IMU的积分模型
    dp += dv*dt + 0.5*dq.matrix()*acc*dt2;
    dv += dq.matrix()*acc*dt;
    Eigen::Matrix3d m3dR = dq.matrix()*dR;
    Eigen::Quaterniond qtmp(m3dR);
    if (qtmp.w()<0)
      qtmp.coeffs() *= -1;
    dq = qtmp.normalized();
    dtime += dt;
    current_time = imu->header.stamp.toSec();
  }
}

// 获取平均加速度
Eigen::Vector3d IMUIntegrator::GetAverageAcc() {
  int i = 0;
  Eigen::Vector3d sum_acc(0, 0, 0);
  for(auto & imu : vimuMsg){
    Eigen::Vector3d acc;
    acc << imu->linear_acceleration.x * gnorm,
           imu->linear_acceleration.y * gnorm,
           imu->linear_acceleration.z * gnorm;
    sum_acc += acc;
    i++;
    if(i > 30) break;
  }
  return sum_acc / i;
}

