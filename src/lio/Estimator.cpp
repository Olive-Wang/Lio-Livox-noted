#include "Estimator/Estimator.h"

//这个cpp主要包含了一个名为Estimator的类的实现。这个类主要用于处理点云数据，包括点到线、点到平面的处理，以及地图的更新等功能

/*
* Estimator类的构造函数
*用于初始化类的成员变量
*这个函数主要初始化了一些点云数据和KD树，以及地图的更新线程，滤波参数等
*/
Estimator::Estimator(const float& filter_corner, const float& filter_surf){
  laserCloudCornerFromLocal.reset(new pcl::PointCloud<PointType>);
  laserCloudSurfFromLocal.reset(new pcl::PointCloud<PointType>);
  laserCloudNonFeatureFromLocal.reset(new pcl::PointCloud<PointType>);
  laserCloudCornerLast.resize(SLIDEWINDOWSIZE);
  for(auto& p:laserCloudCornerLast)
    p.reset(new pcl::PointCloud<PointType>);
  laserCloudSurfLast.resize(SLIDEWINDOWSIZE);
  for(auto& p:laserCloudSurfLast)
    p.reset(new pcl::PointCloud<PointType>);
  laserCloudNonFeatureLast.resize(SLIDEWINDOWSIZE);
  for(auto& p:laserCloudNonFeatureLast)
    p.reset(new pcl::PointCloud<PointType>);
  laserCloudCornerStack.resize(SLIDEWINDOWSIZE);
  for(auto& p:laserCloudCornerStack)
    p.reset(new pcl::PointCloud<PointType>);
  laserCloudSurfStack.resize(SLIDEWINDOWSIZE);
  for(auto& p:laserCloudSurfStack)
    p.reset(new pcl::PointCloud<PointType>);
  laserCloudNonFeatureStack.resize(SLIDEWINDOWSIZE);
  for(auto& p:laserCloudNonFeatureStack)
    p.reset(new pcl::PointCloud<PointType>);
  laserCloudCornerForMap.reset(new pcl::PointCloud<PointType>);
  laserCloudSurfForMap.reset(new pcl::PointCloud<PointType>);
  laserCloudNonFeatureForMap.reset(new pcl::PointCloud<PointType>);
  transformForMap.setIdentity();
  kdtreeCornerFromLocal.reset(new pcl::KdTreeFLANN<PointType>);
  kdtreeSurfFromLocal.reset(new pcl::KdTreeFLANN<PointType>);
  kdtreeNonFeatureFromLocal.reset(new pcl::KdTreeFLANN<PointType>);

  for(int i = 0; i < localMapWindowSize; i++){
    localCornerMap[i].reset(new pcl::PointCloud<PointType>);
    localSurfMap[i].reset(new pcl::PointCloud<PointType>);
    localNonFeatureMap[i].reset(new pcl::PointCloud<PointType>);
  }

  downSizeFilterCorner.setLeafSize(filter_corner, filter_corner, filter_corner);
  downSizeFilterSurf.setLeafSize(filter_surf, filter_surf, filter_surf);
  downSizeFilterNonFeature.setLeafSize(0.4, 0.4, 0.4);
  map_manager = new MAP_MANAGER(filter_corner, filter_surf);
  threadMap = std::thread(&Estimator::threadMapIncrement, this);
}
Estimator::~Estimator(){
  delete map_manager;
}


/*
*这个函数用于在一个新的线程中更新地图。
*这个函数会不断地从laserCloudCornerForMap、laserCloudSurfForMap和laserCloudNonFeatureForMap中获取数据
*然后调用map_manager的featureAssociateToMap和MapIncrement函数来更新地图。
*/
[[noreturn]] void Estimator::threadMapIncrement(){
  pcl::PointCloud<PointType>::Ptr laserCloudCorner(new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr laserCloudSurf(new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr laserCloudNonFeature(new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr laserCloudCorner_to_map(new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr laserCloudSurf_to_map(new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr laserCloudNonFeature_to_map(new pcl::PointCloud<PointType>);
  Eigen::Matrix4d transform;
  while(true){
    std::unique_lock<std::mutex> locker(mtx_Map);
    if(!laserCloudCornerForMap->empty()){

      map_update_ID ++;

      map_manager->featureAssociateToMap(laserCloudCornerForMap,
                                         laserCloudSurfForMap,
                                         laserCloudNonFeatureForMap,
                                         laserCloudCorner,
                                         laserCloudSurf,
                                         laserCloudNonFeature,
                                         transformForMap);
      laserCloudCornerForMap->clear();
      laserCloudSurfForMap->clear();
      laserCloudNonFeatureForMap->clear();
      transform = transformForMap;
      locker.unlock();

      *laserCloudCorner_to_map += *laserCloudCorner;
      *laserCloudSurf_to_map += *laserCloudSurf;
      *laserCloudNonFeature_to_map += *laserCloudNonFeature;

      laserCloudCorner->clear();
      laserCloudSurf->clear();
      laserCloudNonFeature->clear();

      if(map_update_ID % map_skip_frame == 0){
        map_manager->MapIncrement(laserCloudCorner_to_map, 
                                  laserCloudSurf_to_map, 
                                  laserCloudNonFeature_to_map,
                                  transform);

        laserCloudCorner_to_map->clear();
        laserCloudSurf_to_map->clear();
        laserCloudNonFeature_to_map->clear();
      }
      
    }else
      locker.unlock();

    std::chrono::milliseconds dura(2);
    std::this_thread::sleep_for(dura);
  }
}

/**
 * 用ceres构建点线残差的代价函数
  * 这个函数会从laserCloudCorner和laserCloudCornerLocal中获取数据，
 * 然后计算出点到线的特征，
 * 并将这些特征添加到edges和vLineFeatures中
 * @param edges 一个ceres::CostFunction类型的向量，用于存储计算出的点到线的特征。
 * @param vLineFeatures 一个FeatureLine类型的向量，用于存储计算出的点到线的特征。
 * vLineFeatures.emplace_back(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                   tripod1,
                                   tripod2);
 * @param laserCloudCorner 指向全局角点云
 * @param laserCloudCornerLocal  pcl::PointCloud<PointType>::Ptr类型的指针，指向局部的角点云
 * @param kdtreeLocal 一个pcl::KdTreeFLANN<PointType>::Ptr类型的指针，指向局部点云的KD树，用于快速查找最近邻点
 * @param exTlb 表示从激光雷达到body的变换
 * @param m4d 从body到世界的变换
 *  https://blog.csdn.net/p942005405/article/details/125039343
*/
void Estimator::processPointToLine(std::vector<ceres::CostFunction *>& edges,
                                   std::vector<FeatureLine>& vLineFeatures, 
                                   const pcl::PointCloud<PointType>::Ptr& laserCloudCorner,
                                   const pcl::PointCloud<PointType>::Ptr& laserCloudCornerLocal,
                                   const pcl::KdTreeFLANN<PointType>::Ptr& kdtreeLocal,
                                   const Eigen::Matrix4d& exTlb,
                                   const Eigen::Matrix4d& m4d){

/**
 * 这段主要处理对exTlb求逆计算Tbl
*/
  Eigen::Matrix4d Tbl = Eigen::Matrix4d::Identity();
  Tbl.topLeftCorner(3,3) = exTlb.topLeftCorner(3,3).transpose(); //旋转矩阵直接取逆（转置）
  Tbl.topRightCorner(3,1) = -1.0 * Tbl.topLeftCorner(3,3) * exTlb.topRightCorner(3,1);//转置要左乘-R^(-1)

/**
 * 如果vLineFeatures不为空，那么就遍历vLineFeatures
 * 对每个特征创建一个Cost_NavState_IMU_Line类型的代价函数，并添加到edges中。然后函数返回。
*/
  if(!vLineFeatures.empty()){
    for(const auto& l : vLineFeatures){
      auto* e = Cost_NavState_IMU_Line::Create(l.pointOri,
                                               l.lineP1,
                                               l.lineP2,
                                               Tbl,
                                               Eigen::Matrix<double, 1, 1>(1/IMUIntegrator::lidar_m));
      edges.push_back(e);
    }
    return;
  }
  //typedef pcl::PointXYZINormal PointType;
  PointType _pointOri, _pointSel, _coeff;
  std::vector<int> _pointSearchInd;  
  std::vector<float> _pointSearchSqDis; 
  std::vector<int> _pointSearchInd2; 
  std::vector<float> _pointSearchSqDis2;

  Eigen::Matrix< double, 3, 3 > _matA1;
  _matA1.setZero();

/**
 * @param laserCloudCornerStackNum 表示角点云的点数
*/
  int laserCloudCornerStackNum = laserCloudCorner->points.size();
  pcl::PointCloud<PointType>::Ptr kd_pointcloud(new pcl::PointCloud<PointType>);
  int debug_num1 = 0;
  int debug_num2 = 0;
  int debug_num12 = 0;
  int debug_num22 = 0;

  //遍历角点云（当前帧）
  for (int i = 0; i < laserCloudCornerStackNum; i++) {
    _pointOri = laserCloudCorner->points[i];
    //将点从激光雷达坐标系转换到世界坐标系下
    MAP_MANAGER::pointAssociateToMap(&_pointOri, &_pointSel, m4d);
    //在地图中查找最近邻点 
    //!id 是什么？id == 5000是代表距离很远吗？ 索引找不到
    int id = map_manager->FindUsedCornerMap(&_pointSel,laserCenWidth_last,laserCenHeight_last,laserCenDepth_last);
    //如果没有找到最近邻点，那么就跳过这个点
    if(id == 5000) continue;
    //如果点的坐标有问题（NaN），那么就跳过这个点
    if(std::isnan(_pointSel.x) || std::isnan(_pointSel.y) ||std::isnan(_pointSel.z)) continue;
     //如果全局角点云的点数大于100，那么就从全局角点云中查找最近邻点
    //! GlobalCornerMap[i] = map_manager->laserCloudCorner_for_match[i];
    if(GlobalCornerMap[id].points.size() > 100) {
      CornerKdMap[id].nearestKSearch(_pointSel, 5, _pointSearchInd, _pointSearchSqDis);//在地图中查找最近的5个点
      /*
    pcl::KdTreeFLANN<PointT, Dist>::nearestKSearch (const PointT &point, int k, 
                                                    std::vector<int> &k_indices, 
                                                    std::vector<float> &k_distances) const

    //************************************
    Method:    nearestKSearch k-近邻搜索，搜索离point最近的k个点
    注意此方法不对输入索引进行任何检查（即index >= cloud.points.size（） || index < 0），并假定是有效（即有限）数据。
    FullName:  pcl::KdTreeFLANN<PointT, Dist>::nearestKSearch
    Access:    public 
    Returns:   int 返回搜索到的点的个数
    Parameter: const PointT & point 搜索离point最近的k个点
    Parameter: int k 搜索离point最近的k个点
    Parameter: std::vector<int> & k_indices 搜索到的点在数据源中的下标
    Parameter: std::vector<float> & k_distances point到被搜索点的距离，与下标相对应
    //************************************
      */

     //如果最远的点的距离小于阈值
      if (_pointSearchSqDis[4] < thres_dist) {

        debug_num1 ++;
      float cx = 0;
      float cy = 0;
      float cz = 0;
      for (int j = 0; j < 5; j++) {
        cx += GlobalCornerMap[id].points[_pointSearchInd[j]].x; //将最近的5个点的坐标相加
        cy += GlobalCornerMap[id].points[_pointSearchInd[j]].y;
        cz += GlobalCornerMap[id].points[_pointSearchInd[j]].z;
      }
      //算五个点的均值
      cx /= 5;
      cy /= 5;
      cz /= 5;

      float a11 = 0;
      float a12 = 0;
      float a13 = 0;
      float a22 = 0;
      float a23 = 0;
      float a33 = 0;
      for (int j = 0; j < 5; j++) {
        float ax = GlobalCornerMap[id].points[_pointSearchInd[j]].x - cx;//将最近的5个点的坐标减去中心点的坐标
        float ay = GlobalCornerMap[id].points[_pointSearchInd[j]].y - cy;//相当于X-E(X)
        float az = GlobalCornerMap[id].points[_pointSearchInd[j]].z - cz;
        //计算协方差矩阵
        //https://blog.csdn.net/xiaojinger_123/article/details/130749074
        a11 += ax * ax;  //a11 = E((X-E(X))^2)
        a12 += ax * ay;
        a13 += ax * az;
        a22 += ay * ay;
        a23 += ay * az;
        a33 += az * az;
      }
      a11 /= 5;  //a11 = E((X-E(X))^2)/5 = cov(X,X)
      a12 /= 5; //a12 = E((X-E(X))(Y-E(Y)))/5 = cov(X,Y)
      a13 /= 5;
      a22 /= 5;
      a23 /= 5;
      a33 /= 5;
      /*
     * @param   _matA1 协方差矩阵
      */
      _matA1(0, 0) = a11;
      _matA1(0, 1) = a12;
      _matA1(0, 2) = a13;
      _matA1(1, 0) = a12;
      _matA1(1, 1) = a22;
      _matA1(1, 2) = a23;
      _matA1(2, 0) = a13;
      _matA1(2, 1) = a23;
      _matA1(2, 2) = a33;

      /**
        取出特征值和特征向量
          Eigen::VectorXd eigenvalues = eigen_solver.eigenvalues();
          Eigen::MatrixXd eigenvectors = eigen_solver.eigenvectors();
      */
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(_matA1);//计算自伴随矩阵_matA1的特征值和特征向量
      Eigen::Vector3d unit_direction = saes.eigenvectors().col(2); //取最大特征值对应的特征向量（边缘线方向），从小到大排列，第三列是最大的

      if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1]) { //如果最大特征值大于次大特征值的3倍，那么就认为这个点是边缘线点
        debug_num12 ++;
        //计算边缘线的两个点
        float x1 = cx + 0.1 * unit_direction[0];
        float y1 = cy + 0.1 * unit_direction[1];
        float z1 = cz + 0.1 * unit_direction[2];
        float x2 = cx - 0.1 * unit_direction[0];
        float y2 = cy - 0.1 * unit_direction[1];
        float z2 = cz - 0.1 * unit_direction[2];
        
        Eigen::Vector3d tripod1(x1, y1, z1);
        Eigen::Vector3d tripod2(x2, y2, z2);
        //_pointOri = laserCloudCorner->points[i];
        auto* e = Cost_NavState_IMU_Line::Create(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                                 tripod1,
                                                 tripod2,
                                                 Tbl,
                                                 Eigen::Matrix<double, 1, 1>(1/IMUIntegrator::lidar_m));
        edges.push_back(e);
        vLineFeatures.emplace_back(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                   tripod1,
                                   tripod2);
        vLineFeatures.back().ComputeError(m4d); // @param m4d 从body到世界的变换

        continue;
      }
      
    }
    
    }
    //如果局部角点云的点数大于20，那么就从局部角点云中查找最近邻点
    if(laserCloudCornerLocal->points.size() > 20 ){
      kdtreeLocal->nearestKSearch(_pointSel, 5, _pointSearchInd2, _pointSearchSqDis2);
      if (_pointSearchSqDis2[4] < thres_dist) {

        debug_num2 ++;
        float cx = 0;
        float cy = 0;
        float cz = 0;
        for (int j = 0; j < 5; j++) {
          cx += laserCloudCornerLocal->points[_pointSearchInd2[j]].x;
          cy += laserCloudCornerLocal->points[_pointSearchInd2[j]].y;
          cz += laserCloudCornerLocal->points[_pointSearchInd2[j]].z;
        }
        cx /= 5;
        cy /= 5;
        cz /= 5;

        float a11 = 0;
        float a12 = 0;
        float a13 = 0;
        float a22 = 0;
        float a23 = 0;
        float a33 = 0;
        for (int j = 0; j < 5; j++) {
          float ax = laserCloudCornerLocal->points[_pointSearchInd2[j]].x - cx;
          float ay = laserCloudCornerLocal->points[_pointSearchInd2[j]].y - cy;
          float az = laserCloudCornerLocal->points[_pointSearchInd2[j]].z - cz;

          a11 += ax * ax;
          a12 += ax * ay;
          a13 += ax * az;
          a22 += ay * ay;
          a23 += ay * az;
          a33 += az * az;
        }
        a11 /= 5;
        a12 /= 5;
        a13 /= 5;
        a22 /= 5;
        a23 /= 5;
        a33 /= 5;

        _matA1(0, 0) = a11;
        _matA1(0, 1) = a12;
        _matA1(0, 2) = a13;
        _matA1(1, 0) = a12;
        _matA1(1, 1) = a22;
        _matA1(1, 2) = a23;
        _matA1(2, 0) = a13;
        _matA1(2, 1) = a23;
        _matA1(2, 2) = a33;

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(_matA1);
      Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);

        if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1]) {
          debug_num22++;
          float x1 = cx + 0.1 * unit_direction[0];
          float y1 = cy + 0.1 * unit_direction[1];
          float z1 = cz + 0.1 * unit_direction[2];
          float x2 = cx - 0.1 * unit_direction[0];
          float y2 = cy - 0.1 * unit_direction[1];
          float z2 = cz - 0.1 * unit_direction[2];

          Eigen::Vector3d tripod1(x1, y1, z1);
          Eigen::Vector3d tripod2(x2, y2, z2);
          auto* e = Cost_NavState_IMU_Line::Create(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                                  tripod1,
                                                  tripod2,
                                                  Tbl,
                                                  Eigen::Matrix<double, 1, 1>(1/IMUIntegrator::lidar_m));
          edges.push_back(e);
          vLineFeatures.emplace_back(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                    tripod1,
                                    tripod2);
          vLineFeatures.back().ComputeError(m4d);
        }
      }
    }
     
  }
}

/**
 * 这个函数用于处理点到平面的特征。
 * 这个函数会从laserCloudSurf和laserCloudSurfLocal中获取数据，
 * 然后计算出点到平面的特征，并将这些特征添加到edges和vPlanFeatures中
 * https://xiaotaoguo.com/p/paper-note-loam-math/
*/
void Estimator::processPointToPlan(std::vector<ceres::CostFunction *>& edges,
                                   std::vector<FeaturePlan>& vPlanFeatures,
                                   const pcl::PointCloud<PointType>::Ptr& laserCloudSurf,
                                   const pcl::PointCloud<PointType>::Ptr& laserCloudSurfLocal,
                                   const pcl::KdTreeFLANN<PointType>::Ptr& kdtreeLocal,
                                   const Eigen::Matrix4d& exTlb,
                                   const Eigen::Matrix4d& m4d){
  Eigen::Matrix4d Tbl = Eigen::Matrix4d::Identity();
  Tbl.topLeftCorner(3,3) = exTlb.topLeftCorner(3,3).transpose();
  Tbl.topRightCorner(3,1) = -1.0 * Tbl.topLeftCorner(3,3) * exTlb.topRightCorner(3,1);
  if(!vPlanFeatures.empty()){
    for(const auto& p : vPlanFeatures){
      auto* e = Cost_NavState_IMU_Plan::Create(p.pointOri,
                                               p.pa,
                                               p.pb,
                                               p.pc,
                                               p.pd,
                                               Tbl,
                                               Eigen::Matrix<double, 1, 1>(1/IMUIntegrator::lidar_m));
      edges.push_back(e);
    }
    return;
  }
  PointType _pointOri, _pointSel, _coeff;
  std::vector<int> _pointSearchInd;
  std::vector<float> _pointSearchSqDis;
  std::vector<int> _pointSearchInd2;
  std::vector<float> _pointSearchSqDis2;

  Eigen::Matrix< double, 5, 3 > _matA0;
  _matA0.setZero();
  Eigen::Matrix< double, 5, 1 > _matB0;
  _matB0.setOnes();
  _matB0 *= -1;
  Eigen::Matrix< double, 3, 1 > _matX0;
  _matX0.setZero();
  int laserCloudSurfStackNum = laserCloudSurf->points.size();

  int debug_num1 = 0;
  int debug_num2 = 0;
  int debug_num12 = 0;
  int debug_num22 = 0;
  for (int i = 0; i < laserCloudSurfStackNum; i++) {
    _pointOri = laserCloudSurf->points[i];
    MAP_MANAGER::pointAssociateToMap(&_pointOri, &_pointSel, m4d);

    int id = map_manager->FindUsedSurfMap(&_pointSel,laserCenWidth_last,laserCenHeight_last,laserCenDepth_last);
//!
    if(id == 5000) continue;

    if(std::isnan(_pointSel.x) || std::isnan(_pointSel.y) ||std::isnan(_pointSel.z)) continue;

    if(GlobalSurfMap[id].points.size() > 50) {
      SurfKdMap[id].nearestKSearch(_pointSel, 5, _pointSearchInd, _pointSearchSqDis);

      if (_pointSearchSqDis[4] < 1.0) {
        debug_num1 ++;
        /**
         * @param _matA0 Eigen::Matrix< double, 5, 3 > _matA0 用于存储最近的5个点的坐标
        */
        for (int j = 0; j < 5; j++) {
          _matA0(j, 0) = GlobalSurfMap[id].points[_pointSearchInd[j]].x;
          _matA0(j, 1) = GlobalSurfMap[id].points[_pointSearchInd[j]].y;
          _matA0(j, 2) = GlobalSurfMap[id].points[_pointSearchInd[j]].z;
        }
        //colPivHouseholderQr函数对_matA0进行QR分解
        //并使用solve函数将结果存储在_matX0矩阵中。
        _matX0 = _matA0.colPivHouseholderQr().solve(_matB0);
        //计算平面的参数  ax+by+cz+d=0,    (AX=b) A:_matA0 b:_matB0 x:待求法向量
        float pa = _matX0(0, 0);
        float pb = _matX0(1, 0);
        float pc = _matX0(2, 0);
        float pd = 1;
        //将pa,pb,pc,pd归一化
        float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
        pa /= ps;
        pb /= ps;
        pc /= ps;
        pd /= ps;

        bool planeValid = true;
        for (int j = 0; j < 5; j++) {
          //如果点到平面的距离大于阈值，那么就认为这个点不是平面点
          if (std::fabs(pa * GlobalSurfMap[id].points[_pointSearchInd[j]].x +
                        pb * GlobalSurfMap[id].points[_pointSearchInd[j]].y +
                        pc * GlobalSurfMap[id].points[_pointSearchInd[j]].z + pd) > 0.2) {
            planeValid = false;
            break;
          }
        }

        if (planeValid) {
          debug_num12 ++;
          auto* e = Cost_NavState_IMU_Plan::Create(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                                  pa,
                                                  pb,
                                                  pc,
                                                  pd,
                                                  Tbl,
                                                  Eigen::Matrix<double, 1, 1>(1/IMUIntegrator::lidar_m));
          edges.push_back(e);
          vPlanFeatures.emplace_back(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                    pa,
                                    pb,
                                    pc,
                                    pd);
          vPlanFeatures.back().ComputeError(m4d);

          continue;
        }
        
      }
    }
    if(laserCloudSurfLocal->points.size() > 20 ){
    kdtreeLocal->nearestKSearch(_pointSel, 5, _pointSearchInd2, _pointSearchSqDis2);
    if (_pointSearchSqDis2[4] < 1.0) {
      debug_num2++;
      for (int j = 0; j < 5; j++) { 
        _matA0(j, 0) = laserCloudSurfLocal->points[_pointSearchInd2[j]].x;
        _matA0(j, 1) = laserCloudSurfLocal->points[_pointSearchInd2[j]].y;
        _matA0(j, 2) = laserCloudSurfLocal->points[_pointSearchInd2[j]].z;
      }
      _matX0 = _matA0.colPivHouseholderQr().solve(_matB0);

      float pa = _matX0(0, 0);
      float pb = _matX0(1, 0);
      float pc = _matX0(2, 0);
      float pd = 1;

      float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
      pa /= ps;
      pb /= ps;
      pc /= ps;
      pd /= ps;

      bool planeValid = true;
      for (int j = 0; j < 5; j++) {
        if (std::fabs(pa * laserCloudSurfLocal->points[_pointSearchInd2[j]].x +
                      pb * laserCloudSurfLocal->points[_pointSearchInd2[j]].y +
                      pc * laserCloudSurfLocal->points[_pointSearchInd2[j]].z + pd) > 0.2) {
          planeValid = false;
          break;
        }
      }

      if (planeValid) {
        debug_num22 ++;
        auto* e = Cost_NavState_IMU_Plan::Create(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                                pa,
                                                pb,
                                                pc,
                                                pd,
                                                Tbl,
                                                Eigen::Matrix<double, 1, 1>(1/IMUIntegrator::lidar_m));
        edges.push_back(e);
        vPlanFeatures.emplace_back(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                  pa,
                                  pb,
                                  pc,
                                  pd);
        vPlanFeatures.back().ComputeError(m4d);
      }
    }
  }

  }

}

/**
 * 这个函数用于处理点到平面向量的特征。
 * 这个函数会从laserCloudSurf和laserCloudSurfLocal中获取数据，
 * 然后计算出点到平面向量的特征，并将这些特征添加到edges和vPlanFeatures中。
*/
void Estimator::processPointToPlanVec(std::vector<ceres::CostFunction *>& edges,
                                   std::vector<FeaturePlanVec>& vPlanFeatures,
                                   const pcl::PointCloud<PointType>::Ptr& laserCloudSurf,
                                   const pcl::PointCloud<PointType>::Ptr& laserCloudSurfLocal,
                                   const pcl::KdTreeFLANN<PointType>::Ptr& kdtreeLocal,
                                   const Eigen::Matrix4d& exTlb,
                                   const Eigen::Matrix4d& m4d){
  // 初始化转换矩阵
  Eigen::Matrix4d Tbl = Eigen::Matrix4d::Identity();
  // 设置转换矩阵的左上角3x3子矩阵为exTlb的左上角3x3子矩阵的转置
  Tbl.topLeftCorner(3,3) = exTlb.topLeftCorner(3,3).transpose();
  // 设置转换矩阵的右上角3x1子矩阵为exTlb的右上角3x1子矩阵的负值乘以Tbl的左上角3x3子矩阵
  Tbl.topRightCorner(3,1) = -1.0 * Tbl.topLeftCorner(3,3) * exTlb.topRightCorner(3,1);
  // 如果vPlanFeatures不为空
  if(!vPlanFeatures.empty()){
    // 遍历vPlanFeatures
    for(const auto& p : vPlanFeatures){
      // 创建一个Cost_NavState_IMU_Plan_Vec对象，并添加到edges中
      auto* e = Cost_NavState_IMU_Plan_Vec::Create(p.pointOri,
                                                   p.pointProj,
                                                   Tbl,
                                                   p.sqrt_info);
      edges.push_back(e);
    }
    // 结束函数
    return;
  }
  // 初始化点和系数
  PointType _pointOri, _pointSel, _coeff;
  // 初始化搜索索引和搜索距离
  std::vector<int> _pointSearchInd;
  std::vector<float> _pointSearchSqDis;
  std::vector<int> _pointSearchInd2;
  std::vector<float> _pointSearchSqDis2;

  // 初始化矩阵
  Eigen::Matrix< double, 5, 3 > _matA0;
  _matA0.setZero();
  Eigen::Matrix< double, 5, 1 > _matB0;
  _matB0.setOnes();
  _matB0 *= -1;
  Eigen::Matrix< double, 3, 1 > _matX0;
  _matX0.setZero();
  // 获取laserCloudSurf的点数
  int laserCloudSurfStackNum = laserCloudSurf->points.size();

  // 初始化调试变量
  int debug_num1 = 0;
  int debug_num2 = 0;
  int debug_num12 = 0;
  int debug_num22 = 0;
  // 遍历laserCloudSurf的所有点
  for (int i = 0; i < laserCloudSurfStackNum; i++) {
    // 获取当前点
    _pointOri = laserCloudSurf->points[i];
    // 将当前点转换到地图坐标系
    MAP_MANAGER::pointAssociateToMap(&_pointOri, &_pointSel, m4d);

    // 查找当前点在哪个地图块中
    int id = map_manager->FindUsedSurfMap(&_pointSel,laserCenWidth_last,laserCenHeight_last,laserCenDepth_last);

    // 如果id为5000，跳过当前循环
    if(id == 5000) continue;

    // 如果当前点的坐标有任何一个是NaN，跳过当前循环
    if(std::isnan(_pointSel.x) || std::isnan(_pointSel.y) ||std::isnan(_pointSel.z)) continue;

    // 如果当前地图块的点数大于50
    if(GlobalSurfMap[id].points.size() > 50) {
      // 在当前地图块中查找最近的5个点
      SurfKdMap[id].nearestKSearch(_pointSel, 5, _pointSearchInd, _pointSearchSqDis);

      // 如果最远的点的距离小于阈值
      if (_pointSearchSqDis[4] < thres_dist) {
        debug_num1 ++;
        // 将最近的5个点的坐标存储到_matA0中
        for (int j = 0; j < 5; j++) {
          _matA0(j, 0) = GlobalSurfMap[id].points[_pointSearchInd[j]].x;
          _matA0(j, 1) = GlobalSurfMap[id].points[_pointSearchInd[j]].y;
          _matA0(j, 2) = GlobalSurfMap[id].points[_pointSearchInd[j]].z;
        }
        // 使用QR分解求解线性方程组_matA0 * x = _matB0，结果存储在_matX0中
        _matX0 = _matA0.colPivHouseholderQr().solve(_matB0);

        // 从_matX0中获取平面的参数
        float pa = _matX0(0, 0);
        float pb = _matX0(1, 0);
        float pc = _matX0(2, 0);
        float pd = 1;

        // 对平面参数进行归一化
        float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
        pa /= ps;
        pb /= ps;
        pc /= ps;
        pd /= ps;

        // 检查平面是否有效
        bool planeValid = true;
        for (int j = 0; j < 5; j++) {
          // 如果点到平面的距离大于阈值，那么就认为这个点不是平面点
          if (std::fabs(pa * GlobalSurfMap[id].points[_pointSearchInd[j]].x +
                        pb * GlobalSurfMap[id].points[_pointSearchInd[j]].y +
                        pc * GlobalSurfMap[id].points[_pointSearchInd[j]].z + pd) > 0.2) {
            planeValid = false;
            break;
          }
        }

        // 如果平面有效
        if (planeValid) {
          // 增加调试计数器
          debug_num12 ++;
          // 计算点到平面的距离
          double dist = pa * _pointSel.x +
                        pb * _pointSel.y +
                        pc * _pointSel.z + pd;
          // 计算平面的法向量
          Eigen::Vector3d omega(pa, pb, pc);
          // 计算点在平面上的投影点
          Eigen::Vector3d point_proj = Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z) - (dist * omega);
          // 创建一个单位向量
          Eigen::Vector3d e1(1, 0, 0);
          // 计算雅可比矩阵
          Eigen::Matrix3d J = e1 * omega.transpose();
          // 对雅可比矩阵进行奇异值分解
          Eigen::JacobiSVD<Eigen::Matrix3d> svd(J, Eigen::ComputeThinU | Eigen::ComputeThinV);
          // 计算旋转矩阵
          Eigen::Matrix3d R_svd = svd.matrixV() * svd.matrixU().transpose();
          // 计算信息矩阵
          Eigen::Matrix3d info = (1.0/IMUIntegrator::lidar_m) * Eigen::Matrix3d::Identity();
          // 调整信息矩阵的权重
          info(1, 1) *= plan_weight_tan;
          info(2, 2) *= plan_weight_tan;
          // 计算平方根信息矩阵
          Eigen::Matrix3d sqrt_info = info * R_svd.transpose();

          // 创建一个代价函数，并添加到edges中
          auto* e = Cost_NavState_IMU_Plan_Vec::Create(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                                       point_proj,
                                                       Tbl,
                                                       sqrt_info);
          edges.push_back(e);
          // 创建一个特征，并添加到vPlanFeatures中
          vPlanFeatures.emplace_back(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                     point_proj,
                                     sqrt_info);
          // 计算特征的误差
          vPlanFeatures.back().ComputeError(m4d);

          // 跳过当前循环的剩余部分，进入下一次循环
          continue;
        }
        
      }
    }


    // 如果本地点云的点数大于20
    if(laserCloudSurfLocal->points.size() > 20 ){
    // 在本地点云中查找最近的5个点
    kdtreeLocal->nearestKSearch(_pointSel, 5, _pointSearchInd2, _pointSearchSqDis2);
    // 如果最远的点的距离小于阈值
    if (_pointSearchSqDis2[4] < thres_dist) {
      // 增加调试计数器
      debug_num2++;
      // 将最近的5个点的坐标存储到_matA0中
      for (int j = 0; j < 5; j++) { 
        _matA0(j, 0) = laserCloudSurfLocal->points[_pointSearchInd2[j]].x;
        _matA0(j, 1) = laserCloudSurfLocal->points[_pointSearchInd2[j]].y;
        _matA0(j, 2) = laserCloudSurfLocal->points[_pointSearchInd2[j]].z;
      }
      // 使用QR分解求解线性方程组_matA0 * x = _matB0，结果存储在_matX0中
      _matX0 = _matA0.colPivHouseholderQr().solve(_matB0);

      // 从_matX0中获取平面的参数
      float pa = _matX0(0, 0);
      float pb = _matX0(1, 0);
      float pc = _matX0(2, 0);
      float pd = 1;

      // 对平面参数进行归一化
      float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
      pa /= ps;
      pb /= ps;
      pc /= ps;
      pd /= ps;

      // 检查平面是否有效
      bool planeValid = true;
      for (int j = 0; j < 5; j++) {
        // 如果点到平面的距离大于阈值，那么就认为这个点不是平面点
        if (std::fabs(pa * laserCloudSurfLocal->points[_pointSearchInd2[j]].x +
                      pb * laserCloudSurfLocal->points[_pointSearchInd2[j]].y +
                      pc * laserCloudSurfLocal->points[_pointSearchInd2[j]].z + pd) > 0.2) {
          planeValid = false;
          break;
        }
      }

      // 如果平面有效
      if (planeValid) {
        // 增加调试计数器
        debug_num22 ++;
        // 计算点到平面的距离
        double dist = pa * _pointSel.x +
                      pb * _pointSel.y +
                      pc * _pointSel.z + pd;
        // 计算平面的法向量
        Eigen::Vector3d omega(pa, pb, pc);
        // 计算点在平面上的投影点
        Eigen::Vector3d point_proj = Eigen::Vector3d(_pointSel.x,_pointSel.y,_pointSel.z) - (dist * omega);
        // 创建一个单位向量
        Eigen::Vector3d e1(1, 0, 0);
        // 计算雅可比矩阵
        Eigen::Matrix3d J = e1 * omega.transpose();
        // 对雅可比矩阵进行奇异值分解
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(J, Eigen::ComputeThinU | Eigen::ComputeThinV);
        // 计算旋转矩阵
        Eigen::Matrix3d R_svd = svd.matrixV() * svd.matrixU().transpose();
        // 计算信息矩阵
        Eigen::Matrix3d info = (1.0/IMUIntegrator::lidar_m) * Eigen::Matrix3d::Identity();
        // 调整信息矩阵的权重
        info(1, 1) *= plan_weight_tan;
        info(2, 2) *= plan_weight_tan;
        // 计算平方根信息矩阵
        Eigen::Matrix3d sqrt_info = info * R_svd.transpose();

        // 创建一个代价函数，并添加到edges中
        auto* e = Cost_NavState_IMU_Plan_Vec::Create(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                                      point_proj,
                                                      Tbl,
                                                      sqrt_info);
        edges.push_back(e);
        // 创建一个特征，并添加到vPlanFeatures中
        vPlanFeatures.emplace_back(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                    point_proj,
                                    sqrt_info);
        vPlanFeatures.back().ComputeError(m4d);
      }
    }
  }

  }

}


void Estimator::processNonFeatureICP(std::vector<ceres::CostFunction *>& edges,
                                     std::vector<FeatureNon>& vNonFeatures,
                                     const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeature,
                                     const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeatureLocal,
                                     const pcl::KdTreeFLANN<PointType>::Ptr& kdtreeLocal,
                                     const Eigen::Matrix4d& exTlb,
                                     const Eigen::Matrix4d& m4d){
  // 初始化转换矩阵为单位矩阵
  Eigen::Matrix4d Tbl = Eigen::Matrix4d::Identity();
  // 设置转换矩阵的左上角3x3子矩阵为exTlb的转置
  Tbl.topLeftCorner(3,3) = exTlb.topLeftCorner(3,3).transpose();
  // 设置转换矩阵的右上角3x1子矩阵为转换矩阵的左上角3x3子矩阵与exTlb的右上角3x1子矩阵的乘积的负数
  Tbl.topRightCorner(3,1) = -1.0 * Tbl.topLeftCorner(3,3) * exTlb.topRightCorner(3,1);
  // 如果非特征向量不为空
  if(!vNonFeatures.empty()){
    // 遍历非特征向量
    for(const auto& p : vNonFeatures){
      // 创建一个代价函数，并添加到edges中
      auto* e = Cost_NonFeature_ICP::Create(p.pointOri,
                                            p.pa,
                                            p.pb,
                                            p.pc,
                                            p.pd,
                                            Tbl,
                                            Eigen::Matrix<double, 1, 1>(1/IMUIntegrator::lidar_m));
      edges.push_back(e);
    }
    // 结束函数
    return;
  }

  // 初始化点和系数
  PointType _pointOri, _pointSel, _coeff;
  // 初始化搜索索引和搜索距离
  std::vector<int> _pointSearchInd;
  std::vector<float> _pointSearchSqDis;
  std::vector<int> _pointSearchInd2;
  std::vector<float> _pointSearchSqDis2;

  // 初始化矩阵
  Eigen::Matrix< double, 5, 3 > _matA0;
  _matA0.setZero();
  Eigen::Matrix< double, 5, 1 > _matB0;
  _matB0.setOnes();
  _matB0 *= -1;
  Eigen::Matrix< double, 3, 1 > _matX0;
  _matX0.setZero();

  // 获取非特征点云的数量
  int laserCloudNonFeatureStackNum = laserCloudNonFeature->points.size();
  // 遍历非特征点云
  for (int i = 0; i < laserCloudNonFeatureStackNum; i++) {
    // 获取当前点
    _pointOri = laserCloudNonFeature->points[i];
    // 将当前点转换到地图坐标系
    MAP_MANAGER::pointAssociateToMap(&_pointOri, &_pointSel, m4d);
    // 查找当前点在全局非特征地图中的位置
    int id = map_manager->FindUsedNonFeatureMap(&_pointSel,laserCenWidth_last,laserCenHeight_last,laserCenDepth_last);

    // 如果找不到，跳过当前循环
    if(id == 5000) continue;

    // 如果当前点的坐标包含NaN，跳过当前循环
    if(std::isnan(_pointSel.x) || std::isnan(_pointSel.y) ||std::isnan(_pointSel.z)) continue;

    // 如果全局非特征地图中的点数大于100
    if(GlobalNonFeatureMap[id].points.size() > 100) {
      // 在全局非特征地图中查找最近的5个点
      NonFeatureKdMap[id].nearestKSearch(_pointSel, 5, _pointSearchInd, _pointSearchSqDis);
      // 如果最远的点的距离小于阈值
      if (_pointSearchSqDis[4] < 1 * thres_dist) {
        // 将最近的5个点的坐标存储到_matA0中
        for (int j = 0; j < 5; j++) {
          _matA0(j, 0) = GlobalNonFeatureMap[id].points[_pointSearchInd[j]].x;
          _matA0(j, 1) = GlobalNonFeatureMap[id].points[_pointSearchInd[j]].y;
          _matA0(j, 2) = GlobalNonFeatureMap[id].points[_pointSearchInd[j]].z;
        }
        // 使用QR分解求解线性方程组_matA0 * x = _matB0，结果存储在_matX0中
        _matX0 = _matA0.colPivHouseholderQr().solve(_matB0);

        // 从_matX0中获取平面的参数
        float pa = _matX0(0, 0);
        float pb = _matX0(1, 0);
        float pc = _matX0(2, 0);
        float pd = 1;

        // 对平面参数进行归一化
        float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
        pa /= ps;
        pb /= ps;
        pc /= ps;
        pd /= ps;

        // 检查平面是否有效
        bool planeValid = true;
        // 遍历最近的5个点
        for (int j = 0; j < 5; j++) {
          // 如果点到平面的距离大于阈值，那么就认为这个点不是平面点
          if (std::fabs(pa * GlobalNonFeatureMap[id].points[_pointSearchInd[j]].x +
                        pb * GlobalNonFeatureMap[id].points[_pointSearchInd[j]].y +
                        pc * GlobalNonFeatureMap[id].points[_pointSearchInd[j]].z + pd) > 0.2) {
            planeValid = false;
            break;
          }
        }

        // 如果平面有效
        if(planeValid) {

          // 创建一个代价函数，并添加到edges中
          auto* e = Cost_NonFeature_ICP::Create(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                                pa,
                                                pb,
                                                pc,
                                                pd,
                                                Tbl,
                                                Eigen::Matrix<double, 1, 1>(1/IMUIntegrator::lidar_m));
          edges.push_back(e);
          // 创建一个特征，并添加到vNonFeatures中
          vNonFeatures.emplace_back(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                    pa,
                                    pb,
                                    pc,
                                    pd);
          // 计算特征的误差
          vNonFeatures.back().ComputeError(m4d);

          // 跳过当前循环
          continue;
        }
      }
    
    }

    // 如果本地非特征点云的点数大于20
    if(laserCloudNonFeatureLocal->points.size() > 20 ){
      // 在本地非特征点云中查找最近的5个点
      kdtreeLocal->nearestKSearch(_pointSel, 5, _pointSearchInd2, _pointSearchSqDis2);
      // 如果最远的点的距离小于阈值
      if (_pointSearchSqDis2[4] < 1 * thres_dist) {
        // 将最近的5个点的坐标存储到_matA0中
        for (int j = 0; j < 5; j++) { 
          _matA0(j, 0) = laserCloudNonFeatureLocal->points[_pointSearchInd2[j]].x;
          _matA0(j, 1) = laserCloudNonFeatureLocal->points[_pointSearchInd2[j]].y;
          _matA0(j, 2) = laserCloudNonFeatureLocal->points[_pointSearchInd2[j]].z;
        }
        // 使用QR分解求解线性方程组_matA0 * x = _matB0，结果存储在_matX0中
        _matX0 = _matA0.colPivHouseholderQr().solve(_matB0);

        // 从_matX0中获取平面的参数
        float pa = _matX0(0, 0);
        float pb = _matX0(1, 0);
        float pc = _matX0(2, 0);
        float pd = 1;

        // 对平面参数进行归一化
        float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
        pa /= ps;
        pb /= ps;
        pc /= ps;
        pd /= ps;

        // 检查平面是否有效
        bool planeValid = true;
        // 遍历最近的5个点
        for (int j = 0; j < 5; j++) {
          // 如果点到平面的距离大于阈值，那么就认为这个点不是平面点
          if (std::fabs(pa * laserCloudNonFeatureLocal->points[_pointSearchInd2[j]].x +
                        pb * laserCloudNonFeatureLocal->points[_pointSearchInd2[j]].y +
                        pc * laserCloudNonFeatureLocal->points[_pointSearchInd2[j]].z + pd) > 0.2) {
            planeValid = false;
            break;
          }
        }

        // 如果平面有效
        if(planeValid) {

          // 创建一个代价函数，输入为原始点的坐标，平面参数，转换矩阵，以及权重
          auto* e = Cost_NonFeature_ICP::Create(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                                pa,
                                                pb,
                                                pc,
                                                pd,
                                                Tbl,
                                                Eigen::Matrix<double, 1, 1>(1/IMUIntegrator::lidar_m));
          // 将创建的代价函数添加到edges中
          edges.push_back(e);
          // 创建一个特征，输入为原始点的坐标和平面参数
          vNonFeatures.emplace_back(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                    pa,
                                    pb,
                                    pc,
                                    pd);
          // 计算特征的误差
          vNonFeatures.back().ComputeError(m4d);
        }
      }
    }
  }

}

//这个函数将LidarFrame列表中的位姿和速度偏差转换为双精度数组，以便在优化过程中使用。
void Estimator::vector2double(const std::list<LidarFrame>& lidarFrameList){
  int i = 0;
  for(const auto& l : lidarFrameList){
    Eigen::Map<Eigen::Matrix<double, 6, 1>> PR(para_PR[i]);
    PR.segment<3>(0) = l.P;
    PR.segment<3>(3) = Sophus::SO3d(l.Q).log();

    Eigen::Map<Eigen::Matrix<double, 9, 1>> VBias(para_VBias[i]);
    VBias.segment<3>(0) = l.V;
    VBias.segment<3>(3) = l.bg;
    VBias.segment<3>(6) = l.ba;
    i++;
  }
}

//这个函数将优化后的双精度数组转换回LidarFrame列表中的位姿和速度偏差
void Estimator::double2vector(std::list<LidarFrame>& lidarFrameList){
  int i = 0;
  for(auto& l : lidarFrameList){
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> PR(para_PR[i]);
    Eigen::Map<const Eigen::Matrix<double, 9, 1>> VBias(para_VBias[i]);
    l.P = PR.segment<3>(0);
    l.Q = Sophus::SO3d::exp(PR.segment<3>(3)).unit_quaternion();
    l.V = VBias.segment<3>(0);
    l.bg = VBias.segment<3>(3);
    l.ba = VBias.segment<3>(6);
    i++;
  }
}

//这个函数用于估计激光雷达的位姿。它首先将雷达帧列表中的每个雷达帧转换为地图坐标系，然后根据地图中的特征点进行优化
/**
 * @param lidarFrameList
 *boost::shared_ptr<std::list<Estimator::LidarFrame>> lidarFrameList;
 * */
void Estimator::EstimateLidarPose(std::list<LidarFrame>& lidarFrameList,
                           const Eigen::Matrix4d& exTlb,
                           const Eigen::Vector3d& gravity,
                           nav_msgs::Odometry& debugInfo){
  
  Eigen::Matrix3d exRbl = exTlb.topLeftCorner(3,3).transpose(); //body in lidar
  Eigen::Vector3d exPbl = -1.0 * exRbl * exTlb.topRightCorner(3,1);
  Eigen::Matrix4d transformTobeMapped = Eigen::Matrix4d::Identity();
  transformTobeMapped.topLeftCorner(3,3) = lidarFrameList.back().Q * exRbl; //imu in lidar
  transformTobeMapped.topRightCorner(3,1) = lidarFrameList.back().Q * exPbl + lidarFrameList.back().P;

  int laserCloudCornerFromMapNum = map_manager->get_corner_map()->points.size();
  int laserCloudSurfFromMapNum = map_manager->get_surf_map()->points.size();
  int laserCloudCornerFromLocalNum = laserCloudCornerFromLocal->points.size();
  int laserCloudSurfFromLocalNum = laserCloudSurfFromLocal->points.size();
  int stack_count = 0;
  for(const auto& l : lidarFrameList){
    laserCloudCornerLast[stack_count]->clear();
    for(const auto& p : l.laserCloud->points){
      /**
       * 看LidarFeatureExtractor.cpp
       * normal_z为1 表示该点已被选为角点特征点
       * normal_z=2和3 代表是平面特征点和非特征点
       * 这段就是把点云归类
      */
      if(std::fabs(p.normal_z - 1.0) < 1e-5)
        laserCloudCornerLast[stack_count]->push_back(p);
    }
    laserCloudSurfLast[stack_count]->clear();
    for(const auto& p : l.laserCloud->points){
      if(std::fabs(p.normal_z - 2.0) < 1e-5)
        laserCloudSurfLast[stack_count]->push_back(p);
    }

    laserCloudNonFeatureLast[stack_count]->clear();
    for(const auto& p : l.laserCloud->points){
      if(std::fabs(p.normal_z - 3.0) < 1e-5)
        laserCloudNonFeatureLast[stack_count]->push_back(p);
    }

    //这里是对点云进行下采样
    laserCloudCornerStack[stack_count]->clear();
    downSizeFilterCorner.setInputCloud(laserCloudCornerLast[stack_count]);
    downSizeFilterCorner.filter(*laserCloudCornerStack[stack_count]);

    laserCloudSurfStack[stack_count]->clear();
    downSizeFilterSurf.setInputCloud(laserCloudSurfLast[stack_count]);
    downSizeFilterSurf.filter(*laserCloudSurfStack[stack_count]);

    laserCloudNonFeatureStack[stack_count]->clear();
    downSizeFilterNonFeature.setInputCloud(laserCloudNonFeatureLast[stack_count]);
    downSizeFilterNonFeature.filter(*laserCloudNonFeatureStack[stack_count]);
    stack_count++;
  }
  if ( ((laserCloudCornerFromMapNum >= 0 && laserCloudSurfFromMapNum > 100) || 
       (laserCloudCornerFromLocalNum >= 0 && laserCloudSurfFromLocalNum > 100))) {
    Estimate(lidarFrameList, exTlb, gravity);
  }

  // 初始化变换矩阵为单位矩阵
  transformTobeMapped = Eigen::Matrix4d::Identity();
  // 设置变换矩阵的左上角3x3子矩阵为雷达帧列表中第一个雷达帧的旋转矩阵与exRbl的乘积
  transformTobeMapped.topLeftCorner(3,3) = lidarFrameList.front().Q * exRbl;
  // 设置变换矩阵的右上角3x1子矩阵为雷达帧列表中第一个雷达帧的旋转矩阵与exPbl的乘积加上雷达帧的位置
  transformTobeMapped.topRightCorner(3,1) = lidarFrameList.front().Q * exPbl + lidarFrameList.front().P;

  // 创建一个独占锁，保护地图数据的访问
  std::unique_lock<std::mutex> locker(mtx_Map);
  // 将雷达帧列表中第一个雷达帧的角点云赋值给地图的角点云
  *laserCloudCornerForMap = *laserCloudCornerStack[0];
  // 将雷达帧列表中第一个雷达帧的平面点云赋值给地图的平面点云
  *laserCloudSurfForMap = *laserCloudSurfStack[0];
  // 将雷达帧列表中第一个雷达帧的非特征点云赋值给地图的非特征点云
  *laserCloudNonFeatureForMap = *laserCloudNonFeatureStack[0];
  // 设置地图的变换矩阵
  transformForMap = transformTobeMapped;
  // 清空本地的角点云、平面点云和非特征点云
  laserCloudCornerFromLocal->clear();
  laserCloudSurfFromLocal->clear();
  laserCloudNonFeatureFromLocal->clear();
  // 在本地地图中增加新的点云数据
  MapIncrementLocal(laserCloudCornerForMap,laserCloudSurfForMap,laserCloudNonFeatureForMap,transformTobeMapped);
  // 释放锁
  locker.unlock();
}

//这个函数是优化过程的主要部分，它使用Ceres库进行非线性优化，优化的目标是最小化IMU和激光雷达的误差。
void Estimator::Estimate(std::list<LidarFrame>& lidarFrameList,
                         const Eigen::Matrix4d& exTlb,
                         const Eigen::Vector3d& gravity){

  int num_corner_map = 0;
  int num_surf_map = 0;

  static uint32_t frame_count = 0;
  int windowSize = lidarFrameList.size();
  Eigen::Matrix4d transformTobeMapped = Eigen::Matrix4d::Identity();
  Eigen::Matrix3d exRbl = exTlb.topLeftCorner(3,3).transpose();
  Eigen::Vector3d exPbl = -1.0 * exRbl * exTlb.topRightCorner(3,1);
  //将角点，平面点和非特征点存入kd-tree
  kdtreeCornerFromLocal->setInputCloud(laserCloudCornerFromLocal);
  kdtreeSurfFromLocal->setInputCloud(laserCloudSurfFromLocal);
  kdtreeNonFeatureFromLocal->setInputCloud(laserCloudNonFeatureFromLocal);

  std::unique_lock<std::mutex> locker3(map_manager->mtx_MapManager);
  //! 从map_manager中获取地图
  for(int i = 0; i < 4851; i++){
    CornerKdMap[i] = map_manager->getCornerKdMap(i);
    SurfKdMap[i] = map_manager->getSurfKdMap(i);
    NonFeatureKdMap[i] = map_manager->getNonFeatureKdMap(i);

    GlobalSurfMap[i] = map_manager->laserCloudSurf_for_match[i];
    GlobalCornerMap[i] = map_manager->laserCloudCorner_for_match[i];
    GlobalNonFeatureMap[i] = map_manager->laserCloudNonFeature_for_match[i];
  }
  laserCenWidth_last = map_manager->get_laserCloudCenWidth_last();
  laserCenHeight_last = map_manager->get_laserCloudCenHeight_last();
  laserCenDepth_last = map_manager->get_laserCloudCenDepth_last();

  locker3.unlock();

  // store point to line features
  std::vector<std::vector<FeatureLine>> vLineFeatures(windowSize);
  for(auto& v : vLineFeatures){
    v.reserve(2000);
  }

  // store point to plan features
  std::vector<std::vector<FeaturePlanVec>> vPlanFeatures(windowSize);
  for(auto& v : vPlanFeatures){
    v.reserve(2000);
  }

  std::vector<std::vector<FeatureNon>> vNonFeatures(windowSize);
  for(auto& v : vNonFeatures){
    v.reserve(2000);
  }

  if(windowSize == SLIDEWINDOWSIZE) {
    plan_weight_tan = 0.0003; 
    thres_dist = 1.0;
  } else {
    plan_weight_tan = 0.0;
    thres_dist = 25.0;
  }

  // 执行优化过程
  const int max_iters = 5;
  /**
  @param iterOpt 迭代次数
  */
  for(int iterOpt=0; iterOpt<max_iters; ++iterOpt){

    vector2double(lidarFrameList);

    ////create huber loss function
    // 创建Huber损失函数
    ceres::LossFunction* loss_function = NULL;
    loss_function = new ceres::HuberLoss(0.1 / IMUIntegrator::lidar_m); 
    // !如果窗口大小等于滑动窗口大小，则损失函数为NULL
    // 否则，损失函数为Huber损失，其参数为0.1除以激光雷达的测量值
    if(windowSize == SLIDEWINDOWSIZE) {
      loss_function = NULL;
    } else {
      loss_function = new ceres::HuberLoss(0.1 / IMUIntegrator::lidar_m);
    }


    // Problem类
   // https://www.cnblogs.com/vivian187/p/15394000.html
    // 创建Ceres优化问题的选项
    ceres::Problem::Options problem_options;
    // 创建Ceres优化问题
    ceres::Problem problem(problem_options);

    // 添加待优化变量 PVQB
    for(int i=0; i<windowSize; ++i) {
      // 添加位姿参数块
      problem.AddParameterBlock(para_PR[i], 6);
    }

    for(int i=0; i<windowSize; ++i)
      // 添加速度和偏差参数块
      problem.AddParameterBlock(para_VBias[i], 9);

    // 添加IMU CostFunction
    for(int f=1; f<windowSize; ++f){
      auto frame_curr = lidarFrameList.begin();
      std::advance(frame_curr, f);  // frame_curr指向当前帧
      // 1. 添加IMU约束的残差块
      problem.AddResidualBlock(Cost_NavState_PRV_Bias::Create(frame_curr->imuIntegrator,
                                                              const_cast<Eigen::Vector3d&>(gravity),
                                                              Eigen::LLT<Eigen::Matrix<double, 15, 15>>
                                                                      (frame_curr->imuIntegrator.GetCovariance().inverse())
                                                                      .matrixL().transpose()),
                               nullptr,
                               para_PR[f-1],
                               para_VBias[f-1],
                               para_PR[f],
                               para_VBias[f]);
    }

    // 如果存在上一次的边缘化信息
    if (last_marginalization_info){
      // 构造新的边缘化因子
      auto *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
      // 添加边缘化的残差块
      problem.AddResidualBlock(marginalization_factor, nullptr,
                               last_marginalization_parameter_blocks);
    }

    // 记录优化前的位姿
    Eigen::Quaterniond q_before_opti = lidarFrameList.back().Q;
    Eigen::Vector3d t_before_opti = lidarFrameList.back().P;

    // 创建存储线特征、平面特征和非特征的容器
    std::vector<std::vector<ceres::CostFunction *>> edgesLine(windowSize);
    std::vector<std::vector<ceres::CostFunction *>> edgesPlan(windowSize);
    std::vector<std::vector<ceres::CostFunction *>> edgesNon(windowSize);
    // 创建线程数组
    std::thread threads[3];
    // 遍历窗口中的每一帧
    for(int f=0; f<windowSize; ++f) {
      auto frame_curr = lidarFrameList.begin();
      std::advance(frame_curr, f);
      // 初始化变换矩阵
      transformTobeMapped = Eigen::Matrix4d::Identity();
      // 设置变换矩阵的左上角3x3子矩阵为当前帧的旋转矩阵与exRbl的乘积
      transformTobeMapped.topLeftCorner(3,3) = frame_curr->Q * exRbl;
      // 设置变换矩阵的右上角3x1子矩阵为当前帧的旋转矩阵与exPbl的乘积加上当前帧的位置
      transformTobeMapped.topRightCorner(3,1) = frame_curr->Q * exPbl + frame_curr->P;
      // 创建第一个线程，处理角点特征
      threads[0] = std::thread(&Estimator::processPointToLine, this,
                               std::ref(edgesLine[f]),
                               std::ref(vLineFeatures[f]),
                               std::ref(laserCloudCornerStack[f]),
                               std::ref(laserCloudCornerFromLocal),
                               std::ref(kdtreeCornerFromLocal),
                               std::ref(exTlb),
                               std::ref(transformTobeMapped));

      // 创建第二个线程，处理平面特征
      threads[1] = std::thread(&Estimator::processPointToPlanVec, this,
                               std::ref(edgesPlan[f]),
                               std::ref(vPlanFeatures[f]),
                               std::ref(laserCloudSurfStack[f]),
                               std::ref(laserCloudSurfFromLocal),
                               std::ref(kdtreeSurfFromLocal),
                               std::ref(exTlb),
                               std::ref(transformTobeMapped));

      // 创建第三个线程，处理非特征点
      threads[2] = std::thread(&Estimator::processNonFeatureICP, this,
                               std::ref(edgesNon[f]),
                               std::ref(vNonFeatures[f]),
                               std::ref(laserCloudNonFeatureStack[f]),
                               std::ref(laserCloudNonFeatureFromLocal),
                               std::ref(kdtreeNonFeatureFromLocal),
                               std::ref(exTlb),
                               std::ref(transformTobeMapped));

      // 等待所有线程完成
      threads[0].join();
      threads[1].join();
      threads[2].join();
    }

    int cntSurf = 0;
    int cntCorner = 0;
    int cntNon = 0;
    //! 为什么要判断windowSize == SLIDEWINDOWSIZE？
    if(windowSize == SLIDEWINDOWSIZE) {
      thres_dist = 1.0;
      /**
       * @param iterOpt 优化迭代次数
      */
      if(iterOpt == 0){
        for(int f=0; f<windowSize; ++f){
           int cntFtu = 0;//特征点计数
           //遍历edgesLine
          for (auto &e : edgesLine[f]) {
            //如果误差大于0，就加入到优化问题中
            if(std::fabs(vLineFeatures[f][cntFtu].error) > 1e-5){
              problem.AddResidualBlock(e, loss_function, para_PR[f]);
              vLineFeatures[f][cntFtu].valid = true;
            }else{
              vLineFeatures[f][cntFtu].valid = false;
            }
            cntFtu++;
            cntCorner++;
          }

          cntFtu = 0;
          //遍历edgesPlan
          for (auto &e : edgesPlan[f]) {
            if(std::fabs(vPlanFeatures[f][cntFtu].error) > 1e-5){
              problem.AddResidualBlock(e, loss_function, para_PR[f]);
              vPlanFeatures[f][cntFtu].valid = true;
            }else{
              vPlanFeatures[f][cntFtu].valid = false;
            }
            cntFtu++;
            cntSurf++;
          }

          //遍历edgesNon
          cntFtu = 0;
          for (auto &e : edgesNon[f]) {
            if(std::fabs(vNonFeatures[f][cntFtu].error) > 1e-5){
              problem.AddResidualBlock(e, loss_function, para_PR[f]);
              vNonFeatures[f][cntFtu].valid = true;
            }else{
              vNonFeatures[f][cntFtu].valid = false;
            }
            cntFtu++;
            cntNon++;
          }
        }
      }else{
        for(int f=0; f<windowSize; ++f){
          int cntFtu = 0;
          for (auto &e : edgesLine[f]) {
            if(vLineFeatures[f][cntFtu].valid) {
              problem.AddResidualBlock(e, loss_function, para_PR[f]);
            }
            cntFtu++;
            cntCorner++;
          }
          cntFtu = 0;
          for (auto &e : edgesPlan[f]) {
            if(vPlanFeatures[f][cntFtu].valid){
              problem.AddResidualBlock(e, loss_function, para_PR[f]);
            }
            cntFtu++;
            cntSurf++;
          }

          cntFtu = 0;
          for (auto &e : edgesNon[f]) {
            if(vNonFeatures[f][cntFtu].valid){
              problem.AddResidualBlock(e, loss_function, para_PR[f]);
            }
            cntFtu++;
            cntNon++;
          }
        }
      }
    } else {
        // 在第一次迭代时，设置优化阈值为10
        if(iterOpt == 0) {
          thres_dist = 10.0; 
        } else {
          // 在后续迭代中，设置优化阈值为1
          thres_dist = 1.0;
        }
        // 遍历窗口中的每一帧
        for(int f=0; f<windowSize; ++f){
          int cntFtu = 0;
          // 遍历每一帧中的线特征
          for (auto &e : edgesLine[f]) {
            // 如果特征的误差大于阈值，则将其添加到优化问题中，并标记为有效
            if(std::fabs(vLineFeatures[f][cntFtu].error) > 1e-5){
              problem.AddResidualBlock(e, loss_function, para_PR[f]);
              vLineFeatures[f][cntFtu].valid = true;
            }else{
              // 否则，标记特征为无效
              vLineFeatures[f][cntFtu].valid = false;
            }
            cntFtu++;
            cntCorner++;
          }
          cntFtu = 0;
          // 遍历每一帧中的平面特征
          for (auto &e : edgesPlan[f]) {
            // 如果特征的误差大于阈值，则将其添加到优化问题中，并标记为有效
            if(std::fabs(vPlanFeatures[f][cntFtu].error) > 1e-5){
              problem.AddResidualBlock(e, loss_function, para_PR[f]);
              vPlanFeatures[f][cntFtu].valid = true;
            }else{
              // 否则，标记特征为无效
              vPlanFeatures[f][cntFtu].valid = false;
            }
            cntFtu++;
            cntSurf++;
          }

          cntFtu = 0;
          // 遍历每一帧中的非特征点
          for (auto &e : edgesNon[f]) {
            // 如果特征的误差大于阈值，则将其添加到优化问题中，并标记为有效
            if(std::fabs(vNonFeatures[f][cntFtu].error) > 1e-5){
              problem.AddResidualBlock(e, loss_function, para_PR[f]);
              vNonFeatures[f][cntFtu].valid = true;
            }else{
              // 否则，标记特征为无效
              vNonFeatures[f][cntFtu].valid = false;
            }
            cntFtu++;
            cntNon++;
          }
        }

    // 设置Ceres求解器的选项
    ceres::Solver::Options options;
    // 使用稠密的Schur补作为线性求解器类型
    options.linear_solver_type = ceres::DENSE_SCHUR;
    // 使用Dogleg策略作为信任区域策略类型
    options.trust_region_strategy_type = ceres::DOGLEG;
    // 设置最大迭代次数为10
    options.max_num_iterations = 10;
    // 不将优化过程输出到标准输出
    options.minimizer_progress_to_stdout = false;
    // 设置优化过程的线程数为6
    options.num_threads = 6;
    // 创建一个求解器摘要来保存优化过程的信息
    ceres::Solver::Summary summary;
    // 使用上述设置的选项和问题来运行求解器
    ceres::Solve(options, &problem, &summary);

    double2vector(lidarFrameList);

    // 获取优化后的位姿
    Eigen::Quaterniond q_after_opti = lidarFrameList.back().Q;
    Eigen::Vector3d t_after_opti = lidarFrameList.back().P;
    // 获取优化后的速度
    Eigen::Vector3d V_after_opti = lidarFrameList.back().V;
    // 计算优化前后位姿的旋转和平移差异
    double deltaR = (q_before_opti.angularDistance(q_after_opti)) * 180.0 / M_PI;
    double deltaT = (t_before_opti - t_after_opti).norm();

    // 如果旋转和平移的差异小于阈值，或者已经达到最大迭代次数，则结束优化
    if (deltaR < 0.05 && deltaT < 0.05 || (iterOpt+1) == max_iters){
      // 输出当前帧的编号
      ROS_INFO("Frame: %d\n",frame_count++);
      // 如果窗口大小不等于滑动窗口大小，则跳出循环
      if(windowSize != SLIDEWINDOWSIZE) break;
      // 应用边缘化
      auto *marginalization_info = new MarginalizationInfo();
      // 如果存在上一次的边缘化信息
      if (last_marginalization_info){
        std::vector<int> drop_set;
        // 遍历上一次边缘化的参数块
        for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
        {
          // 如果参数块是位姿或速度和偏差，则将其添加到drop_set中
          if (last_marginalization_parameter_blocks[i] == para_PR[0] ||
              last_marginalization_parameter_blocks[i] == para_VBias[0])
            drop_set.push_back(i);
        }

        // 创建新的边缘化因子
        auto *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        // 创建新的残差块信息，并添加到marginalization_info中
        auto *residual_block_info = new ResidualBlockInfo(marginalization_factor, nullptr,
                                                          last_marginalization_parameter_blocks,
                                                          drop_set);
        marginalization_info->addResidualBlockInfo(residual_block_info);
      }
      
      // 获取lidarFrameList的迭代器，并将其向前移动一位
      auto frame_curr = lidarFrameList.begin();
      std::advance(frame_curr, 1);
      // 创建IMU的代价函数
      ceres::CostFunction* IMU_Cost = Cost_NavState_PRV_Bias::Create(frame_curr->imuIntegrator,
                                                                     const_cast<Eigen::Vector3d&>(gravity),
                                                                     Eigen::LLT<Eigen::Matrix<double, 15, 15>>
                                                                             (frame_curr->imuIntegrator.GetCovariance().inverse())
                                                                             .matrixL().transpose());
      // 创建残差块信息，并添加到marginalization_info中
      auto *residual_block_info = new ResidualBlockInfo(IMU_Cost, nullptr,
                                                        std::vector<double *>{para_PR[0], para_VBias[0], para_PR[1], para_VBias[1]},
                                                        std::vector<int>{0, 1});
      marginalization_info->addResidualBlockInfo(residual_block_info);

      // 初始化帧索引
      int f = 0;
      // 初始化变换矩阵为单位矩阵
      transformTobeMapped = Eigen::Matrix4d::Identity();
      // 设置变换矩阵的左上角3x3子矩阵为当前帧的旋转矩阵与exRbl的乘积
      transformTobeMapped.topLeftCorner(3,3) = frame_curr->Q * exRbl;
      // 设置变换矩阵的右上角3x1子矩阵为当前帧的旋转矩阵与exPbl的乘积加上当前帧的位置
      transformTobeMapped.topRightCorner(3,1) = frame_curr->Q * exPbl + frame_curr->P;
      // 清空线特征、平面特征和非特征的容器
      edgesLine[f].clear();
      edgesPlan[f].clear();
      edgesNon[f].clear();
      // 创建第一个线程，处理角点特征
      threads[0] = std::thread(&Estimator::processPointToLine, this,
                               std::ref(edgesLine[f]),
                               std::ref(vLineFeatures[f]),
                               std::ref(laserCloudCornerStack[f]),
                               std::ref(laserCloudCornerFromLocal),
                               std::ref(kdtreeCornerFromLocal),
                               std::ref(exTlb),
                               std::ref(transformTobeMapped));

      // 创建第二个线程，处理平面特征
      threads[1] = std::thread(&Estimator::processPointToPlanVec, this,
                               std::ref(edgesPlan[f]),
                               std::ref(vPlanFeatures[f]),
                               std::ref(laserCloudSurfStack[f]),
                               std::ref(laserCloudSurfFromLocal),
                               std::ref(kdtreeSurfFromLocal),
                               std::ref(exTlb),
                               std::ref(transformTobeMapped));

      // 创建第三个线程，处理非特征点
      threads[2] = std::thread(&Estimator::processNonFeatureICP, this,
                               std::ref(edgesNon[f]),
                               std::ref(vNonFeatures[f]),
                               std::ref(laserCloudNonFeatureStack[f]),
                               std::ref(laserCloudNonFeatureFromLocal),
                               std::ref(kdtreeNonFeatureFromLocal),
                               std::ref(exTlb),
                               std::ref(transformTobeMapped));      
                      
      // 等待所有线程完成
      threads[0].join();
      threads[1].join();
      threads[2].join();
      // 初始化特征点计数
      int cntFtu = 0;
      // 遍历线特征
      for (auto &e : edgesLine[f]) {
        // 如果特征有效，则创建残差块信息，并添加到marginalization_info中
        if(vLineFeatures[f][cntFtu].valid){
          auto *residual_block_info = new ResidualBlockInfo(e, nullptr,
                                                            std::vector<double *>{para_PR[0]},
                                                            std::vector<int>{0});
          marginalization_info->addResidualBlockInfo(residual_block_info);
        }
        // 更新特征点计数
        cntFtu++;
      }
      // 重置特征点计数
      cntFtu = 0;
      // 遍历平面特征
      // 遍历平面特征
      for (auto &e : edgesPlan[f]) {
        // 如果特征有效，则创建残差块信息，并添加到marginalization_info中
        if(vPlanFeatures[f][cntFtu].valid){
          auto *residual_block_info = new ResidualBlockInfo(e, nullptr,
                                                            std::vector<double *>{para_PR[0]},
                                                            std::vector<int>{0});
          marginalization_info->addResidualBlockInfo(residual_block_info);
        }
        // 更新特征点计数
        cntFtu++;
      }

      // 重置特征点计数
      cntFtu = 0;
      // 遍历非特征点
      for (auto &e : edgesNon[f]) {
        // 如果特征有效，则创建残差块信息，并添加到marginalization_info中
        if(vNonFeatures[f][cntFtu].valid){
          auto *residual_block_info = new ResidualBlockInfo(e, nullptr,
                                                            std::vector<double *>{para_PR[0]},
                                                            std::vector<int>{0});
          marginalization_info->addResidualBlockInfo(residual_block_info);
        }
        // 更新特征点计数
        cntFtu++;
      }

      // 预边缘化
      marginalization_info->preMarginalize();
      // 边缘化
      marginalization_info->marginalize();

      // 创建一个地址偏移的哈希表
      std::unordered_map<long, double *> addr_shift;
      // 遍历滑动窗口中的每一帧
      for (int i = 1; i < SLIDEWINDOWSIZE; i++)
      {
        // 将当前帧的位姿和速度偏差的地址映射到前一帧的位姿和速度偏差的地址
        addr_shift[reinterpret_cast<long>(para_PR[i])] = para_PR[i - 1];
        addr_shift[reinterpret_cast<long>(para_VBias[i])] = para_VBias[i - 1];
      }
      // 获取边缘化信息的参数块
      std::vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

      // 删除上一次的边缘化信息
      delete last_marginalization_info;
      // 更新边缘化信息和参数块
      last_marginalization_info = marginalization_info;
      last_marginalization_parameter_blocks = parameter_blocks;
      // 跳出循环
      break;
    }

    // 如果窗口大小不等于滑动窗口大小
    if(windowSize != SLIDEWINDOWSIZE) {
      // 遍历窗口中的每一帧
      for(int f=0; f<windowSize; ++f){
        // 清空线特征、平面特征和非特征的容器
        edgesLine[f].clear();
        edgesPlan[f].clear();
        edgesNon[f].clear();
        vLineFeatures[f].clear();
        vPlanFeatures[f].clear();
        vNonFeatures[f].clear();
      }
    }
  }

}
                         }
//这个函数用于更新局部地图。它将新的雷达帧添加到局部地图中，并使用下采样滤波器对地图进行滤波，以保持地图的大小在可管理的范围内。
void Estimator::MapIncrementLocal(const pcl::PointCloud<PointType>::Ptr& laserCloudCornerStack,
                                  const pcl::PointCloud<PointType>::Ptr& laserCloudSurfStack,
                                  const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeatureStack,
                                  const Eigen::Matrix4d& transformTobeMapped){
  int laserCloudCornerStackNum = laserCloudCornerStack->points.size();
  int laserCloudSurfStackNum = laserCloudSurfStack->points.size();
  int laserCloudNonFeatureStackNum = laserCloudNonFeatureStack->points.size();
  PointType pointSel;
  PointType pointSel2;
  size_t Id = localMapID % localMapWindowSize;
  localCornerMap[Id]->clear();
  localSurfMap[Id]->clear();
  localNonFeatureMap[Id]->clear();
  //将角点添加到地图中
  for (int i = 0; i < laserCloudCornerStackNum; i++) {
    MAP_MANAGER::pointAssociateToMap(&laserCloudCornerStack->points[i], &pointSel, transformTobeMapped);
    localCornerMap[Id]->push_back(pointSel);
  }
  //将平面点添加到地图中
  for (int i = 0; i < laserCloudSurfStackNum; i++) {
    MAP_MANAGER::pointAssociateToMap(&laserCloudSurfStack->points[i], &pointSel2, transformTobeMapped);
    localSurfMap[Id]->push_back(pointSel2);
  }
  //将非特征点添加到地图中
  for (int i = 0; i < laserCloudNonFeatureStackNum; i++) {
    MAP_MANAGER::pointAssociateToMap(&laserCloudNonFeatureStack->points[i], &pointSel2, transformTobeMapped);
    localNonFeatureMap[Id]->push_back(pointSel2);
  }
 //
  for (int i = 0; i < localMapWindowSize; i++) {
    *laserCloudCornerFromLocal += *localCornerMap[i];
    *laserCloudSurfFromLocal += *localSurfMap[i];
    *laserCloudNonFeatureFromLocal += *localNonFeatureMap[i];
  }
  pcl::PointCloud<PointType>::Ptr temp(new pcl::PointCloud<PointType>());
  downSizeFilterCorner.setInputCloud(laserCloudCornerFromLocal);
  downSizeFilterCorner.filter(*temp);
  laserCloudCornerFromLocal = temp;
  pcl::PointCloud<PointType>::Ptr temp2(new pcl::PointCloud<PointType>());
  downSizeFilterSurf.setInputCloud(laserCloudSurfFromLocal);
  downSizeFilterSurf.filter(*temp2);
  laserCloudSurfFromLocal = temp2;
  pcl::PointCloud<PointType>::Ptr temp3(new pcl::PointCloud<PointType>());
  downSizeFilterNonFeature.setInputCloud(laserCloudNonFeatureFromLocal);
  downSizeFilterNonFeature.filter(*temp3);
  laserCloudNonFeatureFromLocal = temp3;
  localMapID ++;
}