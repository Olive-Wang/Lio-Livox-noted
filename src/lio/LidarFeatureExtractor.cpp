#include "LidarFeatureExtractor/LidarFeatureExtractor.h"

//构造函数
LidarFeatureExtractor::LidarFeatureExtractor(int n_scans,int NumCurvSize,float DistanceFaraway,int NumFlat,
                                             int PartNum,float FlatThreshold,float BreakCornerDis,float LidarNearestDis,float KdTreeCornerOutlierDis)
                                             :N_SCANS(n_scans),//lidar扫描线数
                                              thNumCurvSize(NumCurvSize),//thNumCurvSize代表计算曲率时，被计算点两端要分别采用几个点
                                              thDistanceFaraway(DistanceFaraway),//判断是否为远点的阈值
                                              thNumFlat(NumFlat),//一个区域内最大允许的的平面点数量，也有可能选出来的平面点数量比thNumFlat大
                                              thPartNum(PartNum),//一次扫描平均分成几个区域，每个区域中对平面特征点（平面点）/角点特征点（边缘点）的数目有限制
                                              thFlatThreshold(FlatThreshold),
                                              thBreakCornerDis(BreakCornerDis),//判断是否为break point的阈值
                                              thLidarNearestDis(LidarNearestDis)//距离小于这个值，不要
                                              {
  //为vlines、vcorner和vsurf这三个成员变量分配内存空间。
  //原始点云数据，vector<点云指针>    vlines.resize(N_SCANS) 将 vlines 初始化为包含 N_SCANS 个元素的空的vector。
  //ctx: PointType - pcl::PointXYZINormal 保存每条线的原始数据，每个元素都是点云指针，代表一条lidar线上的点云
  vlines.resize(N_SCANS);
  //遍历点云
  for(auto & ptr : vlines){
    ptr.reset(new pcl::PointCloud<PointType>());
  }
  //保存每条线的角点的索引 vector<vector<int>>
  vcorner.resize(N_SCANS);
  //保存每条线的surf特征点的索引 vector<vector<int>>s
  vsurf.resize(N_SCANS);
}


/*原理：
在点云数据中，如果所有点都位于一个平面上，则该平面可以用一个二维平面表示。在二维平面中，只需要两个坐标轴（通常是X和Y轴）来表示所有点的位置。这意味着在这个平面上，
所有点的运动方向或分布主要集中在这两个坐标轴上，而在垂直于这个平面的方向上的运动或分布应该相对较小。

奇异值分解（Singular Value Decomposition，SVD）是一种矩阵分解技术，可以将一个矩阵分解成三个矩阵的乘积：A = U * S * V^T。其中，U和V是正交矩阵，S是一个对角矩阵，对角线上的元素称为奇异值。
在这里，我们通过奇异值分解对计算的平均平方误差矩阵进行分解。平均平方误差矩阵描述了点云数据在三个坐标轴（X、Y、Z）方向上的离散程度。若点云在一个平面上，它们的运动主要集中在平面上的两个方向，
即Z轴方向上的方差较小。

当我们比较第一个奇异值 _matD1(0, 0) 和第二个奇异值 _matD1(1, 0) 的大小时，如果第一个奇异值远小于第二个奇异值，即 _matD1(0, 0) < plane_threshold * _matD1(1, 0) 条件成立，
那么可以认为数据在Z轴方向上的方差较小，而主要集中在一个平面上。这是因为如果数据在一个平面上，则在Z轴方向上的运动或分布相对较小，导致第一个奇异值较小。

因此，通过比较奇异值，我们可以得出点云数据是否主要位于一个平面上的结论。请注意，plane_threshold 是一个阈值参数，用于控制判断的敏感度，其具体取值根据具体应用和点云数据的特性进行调整。*/

//判断给定点云列表是否表示一个平面
bool LidarFeatureExtractor::plane_judge(const std::vector<PointType>& point_list,const int plane_threshold)
{
  //计算给定点云列表 point_list 中所有点的平均坐标 (cx, cy, cz)。
  int num = point_list.size();
  float cx = 0;
  float cy = 0;
  float cz = 0;
  for (int j = 0; j < num; j++) {
    cx += point_list[j].x;
    cy += point_list[j].y;
    cz += point_list[j].z;
  }
  //将结果存储在 cx、cy 和 cz 变量中
  cx /= num;
  cy /= num;
  cz /= num;

  //这部分代码计算了点云列表中点与平均点坐标之间的差值，并用这些差值计算平均平方误差矩阵。
  //该平方误差矩阵是一个3x3的矩阵，其中每个元素表示点与平均点之间的偏差。这些偏差后来被用来判断点云列表是否表示一个平面。
  //mean square error
  float a11 = 0;
  float a12 = 0;
  float a13 = 0;
  float a22 = 0;
  float a23 = 0;
  float a33 = 0;
  //遍历点云
  for (int j = 0; j < num; j++) {
    float ax = point_list[j].x - cx;
    float ay = point_list[j].y - cy;
    float az = point_list[j].z - cz;
    //计算每个点到平均点差值的平方和
    a11 += ax * ax;
    a12 += ax * ay;
    a13 += ax * az;
    a22 += ay * ay;
    a23 += ay * az;
    a33 += az * az;
  }
  //平均平方误差值
  a11 /= num;
  a12 /= num;
  a13 /= num;
  a22 /= num;
  a23 /= num;
  a33 /= num;

  Eigen::Matrix< double, 3, 3 > _matA1;
  _matA1.setZero();
  Eigen::Matrix< double, 3, 1 > _matD1;
  _matD1.setZero();
  Eigen::Matrix< double, 3, 3 > _matV1;
  _matV1.setZero();

  //matA1对称矩阵 协方差矩阵
  _matA1(0, 0) = a11;
  _matA1(0, 1) = a12;
  _matA1(0, 2) = a13;
  _matA1(1, 0) = a12;
  _matA1(1, 1) = a22;
  _matA1(1, 2) = a23;
  _matA1(2, 0) = a13;
  _matA1(2, 1) = a23;
  _matA1(2, 2) = a33;

  //使用JacobiSVD（奇异值分解）对 _matA1 进行分解，得到奇异值和左奇异向量矩阵。
  //左奇异向量是原始矩阵的列向量，而右奇异向量是原始矩阵的行向量。
  /* 其他可能的设置参数(未实验验证)
  Eigen::ComputeFullU and Eigen::ComputeFullV:
  这两个选项用于计算完整的左奇异向量矩阵和右奇异向量矩阵。与之前的Thin选项不同，这些选项将计算完整的奇异向量矩阵，不进行截断，输出完整的矩阵。

  Eigen::AllowOutOfCoreComputation:
  这个选项用于允许JacobiSVD在内存不足时，使用外存储器来计算奇异值分解。当处理非常大的矩阵时，可能会遇到内存不足的问题，设置这个选项可以帮助解决这个问题。

  Eigen::ComputeFullU | Eigen::ComputeFullV:
  这是一个组合选项，用于同时计算完整的左奇异向量矩阵和右奇异向量矩阵。

  Eigen::ComputeThinU | Eigen::ComputeThinV | Eigen::FullPivHouseholderQRPreconditioner:
  这也是一个组合选项，用于计算完整的左奇异向量矩阵和右奇异向量矩阵，并使用FullPivHouseholderQR预处理器来优化计算。

  Eigen::ComputeThinU | Eigen::ComputeThinV | Eigen::ComputeFullU | Eigen::ComputeFullV:
  这是一个非常完整的组合选项，可以同时计算所有可能的结果，包括Thin和Full的左右奇异向量矩阵。*/
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(_matA1, Eigen::ComputeThinU | Eigen::ComputeThinV);
  _matD1 = svd.singularValues();
  _matV1 = svd.matrixU();
  //判断点云列表是否表示一个平面
  //SVD分解找到主要分布方向
  //它比较第一个奇异值和第二个奇异值的大小，如果第一个奇异值远远小于第二个奇异值的 plane_threshold 倍，就返回 true 表示点云列表表示一个平面，否则返回 false 表示不是平面。
  if (_matD1(0, 0) < plane_threshold * _matD1(1, 0)) {
    return true;
  }
  else{
    return false;
  }
}

//点云特征提取
//第一个参数是点云指针的引用，类一般都传引用，应该是避免调用拷贝构造函数，提取的特征点存储在两个vector中
//输出：先添加到laserCloudCorner点云中，再将角点特征点和平面特征点的索引分别添加到pointsLessSharp和pointsLessFlat向量中。
/** \brief 提取特征点，其只保留了不是特别平的的点为面特征，保留了断点和平面交点为角点特征
  * \param[in] cloud: 一条扫描线束上的原始雷达点云
  * \param[in] pointsLessSharp: less sharp点的索引
  * \param[in] pointsLessFlat: less flat点的索引
  */
void LidarFeatureExtractor::detectFeaturePoint(pcl::PointCloud<PointType>::Ptr& cloud,
                                                std::vector<int>& pointsLessSharp,
                                                std::vector<int>& pointsLessFlat){
  /*特征标志：使用固定大小的数组CloudFeatureFlag来存储每个点的标志。标志的值用于将每个点归类如下：
  0：未处理的点（初始值为0）
  1：表示当前点与前后点夹角（原点到当前点向量与当前点到前后点的向量的夹角）余弦值均很大；  表示该点前后2or3个点中有3；
  2：曲率最小的几个被标为3的点，或被标为3的远点，或点与前后点的角度小于15°的点；最终存为平面点
  3：所有曲率小于阈值的点，注意：一部分3点后面会变成2点；
  300：强度变化大于阈值且曲率很小的3个点
  150：两个面的夹角构成的边缘点
  100：断点，左右相邻点距离差超过1m,扫描线与邻点向量夹角大于18 100和150的为角点*/

  //用于存储每个点的特征标志的数组，初始时都设置为0。
  int CloudFeatureFlag[20000];

  //用于存储每个点的曲率值的数组。
  float cloudCurvature[20000];

  //用于存储每个点到激光雷达的距离的数组。
  float cloudDepth[20000];

  //用于存储点云点的索引，将会根据曲率值从小到大进行排序。
  int cloudSortInd[20000];

  //用于存储每个点的反射强度的数组。
  float cloudReflect[20000];

  //用于存储点云点的索引，将会根据反射强度从小到大进行排序。
  int reflectSortInd[20000];

  //如果点与前后点的角度均小于15°，将该点标志置为1 
  int cloudAngle[20000];

  //给传进来的点云指针创建一个引用
  pcl::PointCloud<PointType>::Ptr& laserCloudIn = cloud;
  //获取输入点云的大小
  int cloudSize = laserCloudIn->points.size();
  //临时变量point 一个点
  PointType point;
  //定义一个新的点云指针_laserCloud，用于存储去除非法点后的点云，非法点指的是三位坐标含有NaN Inf的值，下面马上就能看出来
  //new出一个点云，()的意思是调用默认构造函数，返回一个点云指针，作为pcl::PointCloud<PointType>::Ptr类构造函数的参数 猜测pcl::PointCloud<PointType>::Ptr类是智能指针类
  pcl::PointCloud<PointType>::Ptr _laserCloud(new pcl::PointCloud<PointType>());
  //预先分配空间，以避免在添加点时不断重新分配内存。
  _laserCloud->reserve(cloudSize);

  //遍历点云中的点
  for (int i = 0; i < cloudSize; i++) {
    point.x = laserCloudIn->points[i].x;
    point.y = laserCloudIn->points[i].y;
    point.z = laserCloudIn->points[i].z;
//条件编译，如果定义了UNDISTORT宏，则执行#ifdef UNDISTORT和#endif之间的代码，否则执行#else和#endif之间的代码。
//如果UNDISTORT被定义，则获取点云中第i个点的normal_x（要看前面处理中传进来的点云normal_x给的是什么数据）。如果未定义UNDISTORT，则将点的法线normal_x设置为1.0（默认值）。
#ifdef UNDISTORT
    point.normal_x = laserCloudIn.points[i].normal_x;
#else
    point.normal_x = 1.0;
#endif

    //获取点云中第i个点的反射强度。
    point.intensity = laserCloudIn->points[i].intensity;
    //如果点的x、y或z坐标中存在非有限值（例如NaN或Inf），则跳过当前点，不加入新的点云对象。
    if (!pcl_isfinite(point.x) ||
        !pcl_isfinite(point.y) ||
        !pcl_isfinite(point.z)) {
      continue;
    }
    //将新的点point加入_laserCloud点云对象中
    _laserCloud->push_back(point);
    //为当前点设置初始特征标志为0（未处理）。
    CloudFeatureFlag[i] = 0;
  }//点云中的点遍历完成一次

  //点云中的点遍历完成后，更新cloudSize，表示去除非法点后的点云大小。，后面点云大小用更新后的，点云就用去掉非法点后的点云
  cloudSize = _laserCloud->size();

  //定义并初始化用于调试的计数器变量。作用后面会看到，这里需要标一下？
  int debugnum1 = 0;
  int debugnum2 = 0;
  int debugnum3 = 0;
  int debugnum4 = 0;
  int debugnum5 = 0;

  //用于控制点云遍历时的步长，默认为1。
  int count_num = 1;
  //用于标记一个点左右侧是否是平面
  bool left_surf_flag = false;
  bool right_surf_flag = false;

  //---------------------------------------- surf feature extract ---------------------------------------------
  /*平面特征提取（surf feature extract）：代码迭代遍历点云以检测平面特征点。它计算每个点的曲率，以确定是否属于平面特征点。该算法考虑点的深度和邻近点之间的角度来分类表面点。*/
  //定义了点云处理的起始和结束索引，排除了前5个点和后5个点。
  //对于平面点的特征有四种标志为1，2，3，300，平面特征提取把一条扫描线分成了100段进行提取，最终选取标志位为2的点为平面点
  int scanStartInd = 5;
  int scanEndInd = cloudSize - 6;

  //用于记录距离过远的点的数量
  int thDistanceFaraway_fea = 0;
  //遍历从第6个点到倒数第6个点的点云数据。
  for (int i = 5; i < cloudSize - 5; i ++ ) {
    //计算坐标差值
    float diffX = 0;
    float diffY = 0;
    float diffZ = 0;
    
    //计算点的距离（深度）：通过sqrt函数计算点到原点的距离。 _laserCloud是去除非法点后的点云
    float dis = sqrt(_laserCloud->points[i].x * _laserCloud->points[i].x +
                     _laserCloud->points[i].y * _laserCloud->points[i].y +
                     _laserCloud->points[i].z * _laserCloud->points[i].z);
    //定义了三个Eigen::Vector3d对象，分别存储当前点、前一个点和后一个点的(x, y, z)坐标。
    Eigen::Vector3d pt_last(_laserCloud->points[i-1].x, _laserCloud->points[i-1].y, _laserCloud->points[i-1].z);
    Eigen::Vector3d pt_cur(_laserCloud->points[i].x, _laserCloud->points[i].y, _laserCloud->points[i].z);
    Eigen::Vector3d pt_next(_laserCloud->points[i+1].x, _laserCloud->points[i+1].y, _laserCloud->points[i+1].z);
    // a•b=|a||b|cosx  >>>  cosx = a•b/|a||b|
    //计算当前点与前后两个点的夹角余弦值。余弦值为1或者-1，表示夹角0度或者180度，点在同一直线，可能出现LOAM论文中（B）情况不被接受
    double angle_last = (pt_last-pt_cur).dot(pt_cur) / ((pt_last-pt_cur).norm()*pt_cur.norm());
    double angle_next = (pt_next-pt_cur).dot(pt_cur) / ((pt_next-pt_cur).norm()*pt_cur.norm());

    //设置thNumCurvSize的值为2或3，thNumCurvSize代表计算曲率时，被计算点两端要分别采用几个点，为什么要这样的策略来取点数？如果是不可靠的情况，就少取少计算；如果是正常区域就两边多算一个
    //   &&优先级大于||
    if (dis > thDistanceFaraway || (fabs(angle_last) > 0.966 && fabs(angle_next) > 0.966)) {
      thNumCurvSize = 2;
    } else {
      thNumCurvSize = 3;
    }

    if(fabs(angle_last) > 0.966 && fabs(angle_next) > 0.966) {
      //如果当前点同时和前后两点的夹角余弦值的绝对值较大，将对应的cloudAngle标志设置为1。
      cloudAngle[i] = 1;
    }

    //计算特征值：通过对当前点前后若干个点的坐标差值，计算得到该点的曲率值、深度、反射强度等特征
    //初始化变量diffR，用于存储特征值中的反射强度信息。 反射强度信息的意义？
    float diffR = -2 * thNumCurvSize * _laserCloud->points[i].intensity;
    //遍历前后点计算特征值：通过对当前点前后若干个点的坐标差值，计算得到该点的曲率值和反射强度。
    for (int j = 1; j <= thNumCurvSize; ++j) {
      //这里还没有算差值，只是把坐标值累计起来
      //在每次循环中，累加了坐标的差值diffX、diffY、diffZ以及反射强度差值diffR。
      diffX += _laserCloud->points[i - j].x + _laserCloud->points[i + j].x;
      diffY += _laserCloud->points[i - j].y + _laserCloud->points[i + j].y;
      diffZ += _laserCloud->points[i - j].z + _laserCloud->points[i + j].z;
      diffR += _laserCloud->points[i - j].intensity + _laserCloud->points[i + j].intensity;
    }
    //这里才一起算了差值
    //修正之前累加的坐标差值，减去当前点坐标的2倍，用于计算曲率。
    diffX -= 2 * thNumCurvSize * _laserCloud->points[i].x;
    diffY -= 2 * thNumCurvSize * _laserCloud->points[i].y;
    diffZ -= 2 * thNumCurvSize * _laserCloud->points[i].z;

    //将特征值存储：将计算得到的特征值存储在对应的数组cloudCurvature、cloudDepth、cloudSortInd、cloudReflect和reflectSortInd中。
    cloudDepth[i] = dis;//存储深度信息
    cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;// / (2 * thNumCurvSize * dis + 1e-3); 存储曲率信息
    cloudSortInd[i] = i;// 存储排序索引，用于后续对曲率和反射强度进行排序
    cloudReflect[i] = diffR;// 存储反射强度信息
    reflectSortInd[i] = i;// 存储反射强度排序索引，用于后续对反射强度进行排序

  }//结束遍历从第6个点到倒数第6个点的点云数据。

  //循环遍历thPartNum个区域：该部分代码对每个区域内的点进行特征标记和提取。
  for (int j = 0; j < thPartNum; j++) {
    //在这个循环中，首先计算每个区域的起始点索引sp和结束点索引ep，通过将整个点云分成thPartNum个区域，以便逐个区域处理点云数据。
    //scanStarInd是从第6个点云开始
    int sp = scanStartInd + (scanEndInd - scanStartInd) * j / thPartNum;
    //-1是因为要把下一段的开始点去掉，但是会导致没有算最后一个点？
    int ep = scanStartInd + (scanEndInd - scanStartInd) * (j + 1) / thPartNum - 1;

    //对区域内的点进行冒泡排序，分别根据曲率值和反射强度从小到大进行排序，以便后续特征提取过程中选择合适的特征点。
    // sort the curvatures from small to large
    for (int k = sp + 1; k <= ep; k++) {
      for (int l = k; l >= sp + 1; l--) {
        if (cloudCurvature[cloudSortInd[l]] <
            cloudCurvature[cloudSortInd[l - 1]]) {
          //对存储索引的vector排序，注意只对索引排序而不是对曲率或反射强度排序
          int temp = cloudSortInd[l - 1];
          cloudSortInd[l - 1] = cloudSortInd[l];
          cloudSortInd[l] = temp;
        }
      }
    }

    // sort the reflectivity from small to large
    for (int k = sp + 1; k <= ep; k++) {
      for (int l = k; l >= sp + 1; l--) {
        if (cloudReflect[reflectSortInd[l]] <
            cloudReflect[reflectSortInd[l - 1]]) {
          int temp = reflectSortInd[l - 1];
          reflectSortInd[l - 1] = reflectSortInd[l];
          reflectSortInd[l] = temp;
        }
      }
    }

    //记录选取的特征点数量：用于记录在当前区域内已经选取的曲率最小点的数量和曲率最大点的数量。
    int smallestPickedNum = 1;
    int sharpestPickedNum = 1;
    //特征点标记和提取：根据一定的条件，对当前区域内的点进行特征标记和提取。
    for (int k = sp; k <= ep; k++) {
      //从曲率值小的索引开始遍历本区域
      int ind = cloudSortInd[k];
      //如果该点已经被标记为特征点，则跳过
      if (CloudFeatureFlag[ind] != 0) continue;
      
      // 如果当前点的曲率值小于一定阈值，则认为该点可能是平面特征点
      if (cloudCurvature[ind] < thFlatThreshold * cloudDepth[ind] * thFlatThreshold * cloudDepth[ind]) {
        //将该点标记为3
        CloudFeatureFlag[ind] = 3;

        //向后查找一定数量的点，如果这些点的坐标变化不大，则将这些点也标记为平坦特征点（待定）
        for (int l = 1; l <= thNumCurvSize; l++) {
          //前后点之间的坐标差值
          float diffX = _laserCloud->points[ind + l].x -
                        _laserCloud->points[ind + l - 1].x;
          float diffY = _laserCloud->points[ind + l].y -
                        _laserCloud->points[ind + l - 1].y;
          float diffZ = _laserCloud->points[ind + l].z -
                        _laserCloud->points[ind + l - 1].z;
          // 如果坐标变化过大或者当前点的深度超过一定阈值，则停止向后查找
          if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.02 || cloudDepth[ind] > thDistanceFaraway) {
            break;
          }
          // 经过考验后，标记为3的点后面的点标记为1 也可能是平面特征点？？？？？？？？？但是后面没有再利用1这个标志
          CloudFeatureFlag[ind + l] = 1;
        }
        //向前查找一定数量的点，如果这些点的坐标变化不大，则将这些点也标记为平坦特征点（待定）
        for (int l = -1; l >= -thNumCurvSize; l--) {
          float diffX = _laserCloud->points[ind + l].x -
                        _laserCloud->points[ind + l + 1].x;
          float diffY = _laserCloud->points[ind + l].y -
                        _laserCloud->points[ind + l + 1].y;
          float diffZ = _laserCloud->points[ind + l].z -
                        _laserCloud->points[ind + l + 1].z;
          if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.02 || cloudDepth[ind] > thDistanceFaraway) {
            break;
          }

          CloudFeatureFlag[ind + l] = 1;
        }
      }
    }//在这个循环中，对当前区域内的点进行遍历，判断是否满足平坦特征的条件。如果满足条件，将该点及其周围的若干点标记为平坦特征点（待定）这里使用了一定的坐标变化阈值和深度阈值来控制点的选择。
    
    for (int k = sp; k <= ep; k++) {
      //从曲率值小的索引开始遍历本区域
      int ind = cloudSortInd[k];
      //如果该点是最平坦的点，并且在规定的最大平面点数量内，则将其标记为普通特征点
      //曲率最小的几个被标为3的点，或被标为3的远点，或点与前后点的角度小于15°的点？？？？ 最终存为平面点
      if(((CloudFeatureFlag[ind] == 3) && (smallestPickedNum <= thNumFlat)) || 
          ((CloudFeatureFlag[ind] == 3) && (cloudDepth[ind] > thDistanceFaraway)) ||
          cloudAngle[ind] == 1){
        smallestPickedNum ++;
        CloudFeatureFlag[ind] = 2;
        // 如果该点的深度超过一定阈值，则增加计数器
        if(cloudDepth[ind] > thDistanceFaraway) {
          thDistanceFaraway_fea++;
        }
      }

      //从反射强度最小的索引开始遍历本区域
      //300——反射强度大于阈值且曲率很小的3个点
      int idx = reflectSortInd[k];
      if(cloudCurvature[idx] < 0.7 * thFlatThreshold * cloudDepth[idx] * thFlatThreshold * cloudDepth[idx]
         && sharpestPickedNum <= 3 && cloudReflect[idx] > 20.0){
        sharpestPickedNum ++;
        CloudFeatureFlag[idx] = 300;
      }
    }
    
  }//结束循环遍历thPartNum个区域：该部分代码对每个区域内的点进行特征标记和提取。

  //---------------------------------------- line feature where surfaces meet -------------------------------------
  /*线特征提取（line feature where surfaces meet）：代码进一步处理点，以确定表面相遇线上的点。*/
  for (int i = 5; i < cloudSize - 5; i += count_num ) {
    //计算当前点 (x, y, z) 到原点 (0, 0, 0) 的欧几里得距离，即点的深度
    float depth = sqrt(_laserCloud->points[i].x * _laserCloud->points[i].x +
                       _laserCloud->points[i].y * _laserCloud->points[i].y +
                       _laserCloud->points[i].z * _laserCloud->points[i].z);
    //left curvature
    //计算左曲率：计算左侧曲率，通过计算当前点和其前四个相邻点之间的差值来得到。这里涉及到坐标差值的计算，并将差值平方后累加得到左曲率值，存储在变量 left_curvature 中。
    float ldiffX =
            _laserCloud->points[i - 4].x + _laserCloud->points[i - 3].x
            - 4 * _laserCloud->points[i - 2].x
            + _laserCloud->points[i - 1].x + _laserCloud->points[i].x;

    float ldiffY =
            _laserCloud->points[i - 4].y + _laserCloud->points[i - 3].y
            - 4 * _laserCloud->points[i - 2].y
            + _laserCloud->points[i - 1].y + _laserCloud->points[i].y;

    float ldiffZ =
            _laserCloud->points[i - 4].z + _laserCloud->points[i - 3].z
            - 4 * _laserCloud->points[i - 2].z
            + _laserCloud->points[i - 1].z + _laserCloud->points[i].z;

    float left_curvature = ldiffX * ldiffX + ldiffY * ldiffY + ldiffZ * ldiffZ;

    //如果左边曲率小
    if(left_curvature < thFlatThreshold * depth){

      std::vector<PointType> left_list;
      //把左边四个点存left_list
      for(int j = -4; j < 0; j++){
        left_list.push_back(_laserCloud->points[i + j]);
      }
      //把左侧标记为平面
      left_surf_flag = true;
    }
    else{
      //把左侧标记为非平面
      left_surf_flag = false;
    }

    //right curvature
    //右侧同理
    float rdiffX =
            _laserCloud->points[i + 4].x + _laserCloud->points[i + 3].x
            - 4 * _laserCloud->points[i + 2].x
            + _laserCloud->points[i + 1].x + _laserCloud->points[i].x;

    float rdiffY =
            _laserCloud->points[i + 4].y + _laserCloud->points[i + 3].y
            - 4 * _laserCloud->points[i + 2].y
            + _laserCloud->points[i + 1].y + _laserCloud->points[i].y;

    float rdiffZ =
            _laserCloud->points[i + 4].z + _laserCloud->points[i + 3].z
            - 4 * _laserCloud->points[i + 2].z
            + _laserCloud->points[i + 1].z + _laserCloud->points[i].z;

    float right_curvature = rdiffX * rdiffX + rdiffY * rdiffY + rdiffZ * rdiffZ;

    if(right_curvature < thFlatThreshold * depth){
      std::vector<PointType> right_list;
      //如果判断右侧是平面，把右侧四个点存入right_list
      for(int j = 1; j < 5; j++){
        right_list.push_back(_laserCloud->points[i + j]);
      }
      //点云遍历步长改为4，跳过下面四个点的遍历，同时标记右侧为平面
      count_num = 4;
      right_surf_flag = true;
    }
    else{
      //否则点云遍历步长还是1，标记右侧为非平面点
      count_num = 1;
      right_surf_flag = false;
    }

    //calculate the included angle
    //如果当前点左右都是平面，那当前点很有可能是平面交线上的点
    if(left_surf_flag && right_surf_flag){
      debugnum4 ++;

      //初始化两个 Eigen::Vector3d 对象 norm_left 和 norm_right
      Eigen::Vector3d norm_left(0,0,0);
      Eigen::Vector3d norm_right(0,0,0);
      //算左侧四个点三维坐标到当前点差值，归一化后求加权平均，离得越远占比越大
      for(int k = 1;k<5;k++){
        Eigen::Vector3d tmp = Eigen::Vector3d(_laserCloud->points[i - k].x - _laserCloud->points[i].x,
                                              _laserCloud->points[i - k].y - _laserCloud->points[i].y,
                                              _laserCloud->points[i - k].z - _laserCloud->points[i].z);
        tmp.normalize();
        norm_left += (k/10.0)* tmp;
      }
      //算右侧
      for(int k = 1;k<5;k++){
        Eigen::Vector3d tmp = Eigen::Vector3d(_laserCloud->points[i + k].x - _laserCloud->points[i].x,
                                              _laserCloud->points[i + k].y - _laserCloud->points[i].y,
                                              _laserCloud->points[i + k].z - _laserCloud->points[i].z);
        tmp.normalize();
        norm_right += (k/10.0)* tmp;
      }

      //calculate the angle between this group and the previous group
      //算左右平面的夹角余弦
      double cc = fabs( norm_left.dot(norm_right) / (norm_left.norm()*norm_right.norm()) );
      //calculate the maximum distance, the distance cannot be too small
      Eigen::Vector3d last_tmp = Eigen::Vector3d(_laserCloud->points[i - 4].x - _laserCloud->points[i].x,
                                                 _laserCloud->points[i - 4].y - _laserCloud->points[i].y,
                                                 _laserCloud->points[i - 4].z - _laserCloud->points[i].z);
      Eigen::Vector3d current_tmp = Eigen::Vector3d(_laserCloud->points[i + 4].x - _laserCloud->points[i].x,
                                                    _laserCloud->points[i + 4].y - _laserCloud->points[i].y,
                                                    _laserCloud->points[i + 4].z - _laserCloud->points[i].z);
      double last_dis = last_tmp.norm();
      double current_dis = current_tmp.norm();

      //150——两个面交线上的点
      //cc<0.5代表夹角大于45°小于135°，很可能是平面交线的点，同时前后点到当前点的距离不能太小，太小可能没有参考价值，比如只是平面的一个突起？？
      if(cc < 0.5 && last_dis > 0.05 && current_dis > 0.05 ){ //
        debugnum5 ++;
        CloudFeatureFlag[i] = 150;
      }
    }

  }

  //--------------------------------------------------- break points ---------------------------------------------
  for(int i = 5; i < cloudSize - 5; i ++){
    float diff_left[2];
    float diff_right[2];
    //算点云深度
    float depth = sqrt(_laserCloud->points[i].x * _laserCloud->points[i].x +
                       _laserCloud->points[i].y * _laserCloud->points[i].y +
                       _laserCloud->points[i].z * _laserCloud->points[i].z);

    for(int count = 1; count < 3; count++ ){
      //diff_right[0]存储后一个点到当前点的距离差值，diff_right[1]存储往后第二个点到当前点的距离差值
      float diffX1 = _laserCloud->points[i + count].x - _laserCloud->points[i].x;
      float diffY1 = _laserCloud->points[i + count].y - _laserCloud->points[i].y;
      float diffZ1 = _laserCloud->points[i + count].z - _laserCloud->points[i].z;
      diff_right[count - 1] = sqrt(diffX1 * diffX1 + diffY1 * diffY1 + diffZ1 * diffZ1);
      //diff_left[0]存储前一个点到当前点的距离差值，diff_left[1]存储往前第二个点到当前点的距离差值
      float diffX2 = _laserCloud->points[i - count].x - _laserCloud->points[i].x;
      float diffY2 = _laserCloud->points[i - count].y - _laserCloud->points[i].y;
      float diffZ2 = _laserCloud->points[i - count].z - _laserCloud->points[i].z;
      diff_left[count - 1] = sqrt(diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2);
    }
    //depth_right存储后一个点的深度
    float depth_right = sqrt(_laserCloud->points[i + 1].x * _laserCloud->points[i + 1].x +
                             _laserCloud->points[i + 1].y * _laserCloud->points[i + 1].y +
                             _laserCloud->points[i + 1].z * _laserCloud->points[i + 1].z);
    //depth_left存储前一个点的深度
    float depth_left = sqrt(_laserCloud->points[i - 1].x * _laserCloud->points[i - 1].x +
                            _laserCloud->points[i - 1].y * _laserCloud->points[i - 1].y +
                            _laserCloud->points[i - 1].z * _laserCloud->points[i - 1].z);
    
    //如果当前点与前后两个点之间的距离差值的绝对值大于阈值 thBreakCornerDis，则进入条件块，表示可能出现了break point。示意图中的不可靠情况
    if(fabs(diff_right[0] - diff_left[0]) > thBreakCornerDis){
      //如果当前点与后一个点之间的距离大于当前点与前一个点之间的距离，说明当前点可能是一个左侧表面的一部分，需要进行左侧表面的验证
      if(diff_right[0] > diff_left[0]){
        //当前点与左侧相邻点的差值向量。
        Eigen::Vector3d surf_vector = Eigen::Vector3d(_laserCloud->points[i - 1].x - _laserCloud->points[i].x,
                                                      _laserCloud->points[i - 1].y - _laserCloud->points[i].y,
                                                      _laserCloud->points[i - 1].z - _laserCloud->points[i].z);
        //表示当前点的激光雷达坐标
        Eigen::Vector3d lidar_vector = Eigen::Vector3d(_laserCloud->points[i].x,
                                                       _laserCloud->points[i].y,
                                                       _laserCloud->points[i].z);
        //模长
        double left_surf_dis = surf_vector.norm();
        //calculate the angle between the laser direction and the surface
        double cc = fabs( surf_vector.dot(lidar_vector) / (surf_vector.norm()*lidar_vector.norm()) );

        //用于存储左侧相邻点的点云容器
        std::vector<PointType> left_list;
        //用于记录左侧相邻点之间的最小距离，初始值设定为一个较大值
        double min_dis = 10000;
        //用于记录左侧相邻点之间的最大距离，初始值设定为0
        double max_dis = 0;
        //循环遍历左侧的3个相邻点，包括当前点在内，一共4个点，以构成一个左侧窗口
        for(int j = 0; j < 4; j++){   //TODO: change the plane window size and add thin rod support
          left_list.push_back(_laserCloud->points[i - j]);
          Eigen::Vector3d temp_vector = Eigen::Vector3d(_laserCloud->points[i - j].x - _laserCloud->points[i - j - 1].x,
                                                        _laserCloud->points[i - j].y - _laserCloud->points[i - j - 1].y,
                                                        _laserCloud->points[i - j].z - _laserCloud->points[i - j - 1].z);
          //这一句的作用是不算左侧第三个点到左侧第四个点的距离，但是前面把左侧第三个点放入了left_list，用于判断左侧是否为平面
          if(j == 3) break;
          double temp_dis = temp_vector.norm();
          //更新相邻点间的最大和最小距离
          if(temp_dis < min_dis) min_dis = temp_dis;
          if(temp_dis > max_dis) max_dis = temp_dis;
        }
        //判断左侧是否为平面
        bool left_is_plane = plane_judge(left_list,100);

        //cc<0.95表示角度在15°-165°之间，左侧是示意图中的第一种不可靠情况
        if( cc < 0.95 ){//(max_dis < 2*min_dis) && left_surf_dis < 0.05 * depth  && left_is_plane &&
        //因为现在认为左边是平面，所以右侧深度一定更大，这是避免第一种不可靠情况
          if(depth_right > depth_left){
            CloudFeatureFlag[i] = 100;
          }
          else{//右侧点不能太近
            if(depth_right == 0) CloudFeatureFlag[i] = 100;
          }
        }
      }
      else{//右侧的同样的验证

        Eigen::Vector3d surf_vector = Eigen::Vector3d(_laserCloud->points[i + 1].x - _laserCloud->points[i].x,
                                                      _laserCloud->points[i + 1].y - _laserCloud->points[i].y,
                                                      _laserCloud->points[i + 1].z - _laserCloud->points[i].z);
        Eigen::Vector3d lidar_vector = Eigen::Vector3d(_laserCloud->points[i].x,
                                                       _laserCloud->points[i].y,
                                                       _laserCloud->points[i].z);
        double right_surf_dis = surf_vector.norm();
        //calculate the angle between the laser direction and the surface
        double cc = fabs( surf_vector.dot(lidar_vector) / (surf_vector.norm()*lidar_vector.norm()) );

        std::vector<PointType> right_list;
        double min_dis = 10000;
        double max_dis = 0;
        for(int j = 0; j < 4; j++){ //TODO: change the plane window size and add thin rod support
          right_list.push_back(_laserCloud->points[i - j]);
          Eigen::Vector3d temp_vector = Eigen::Vector3d(_laserCloud->points[i + j].x - _laserCloud->points[i + j + 1].x,
                                                        _laserCloud->points[i + j].y - _laserCloud->points[i + j + 1].y,
                                                        _laserCloud->points[i + j].z - _laserCloud->points[i + j + 1].z);

          if(j == 3) break;
          double temp_dis = temp_vector.norm();
          if(temp_dis < min_dis) min_dis = temp_dis;
          if(temp_dis > max_dis) max_dis = temp_dis;
        }
        bool right_is_plane = plane_judge(right_list,100);
        if( cc < 0.95){ //right_is_plane  && (max_dis < 2*min_dis) && right_surf_dis < 0.05 * depth &&

          if(depth_right < depth_left){
            CloudFeatureFlag[i] = 100;
          }
          else{
            if(depth_left == 0) CloudFeatureFlag[i] = 100;
          }
        }
      }
    }

    // break points select
    //如果当前索引i处的CloudFeatureFlag值为100，表示这可能是一个break point
    if(CloudFeatureFlag[i] == 100){
      debugnum2++;
      std::vector<Eigen::Vector3d> front_norms;
      Eigen::Vector3d norm_front(0,0,0);
      Eigen::Vector3d norm_back(0,0,0);

      for(int k = 1;k<4;k++){
        //计算左侧点的深度
        float temp_depth = sqrt(_laserCloud->points[i - k].x * _laserCloud->points[i - k].x +
                        _laserCloud->points[i - k].y * _laserCloud->points[i - k].y +
                        _laserCloud->points[i - k].z * _laserCloud->points[i - k].z);
        //如果深度小于1，那么跳过本次循环，继续算下一个左侧点 为什么？？
        if(temp_depth < 1){
          continue;
        }
        //左侧点到当前点的坐标差
        Eigen::Vector3d tmp = Eigen::Vector3d(_laserCloud->points[i - k].x - _laserCloud->points[i].x,
                                              _laserCloud->points[i - k].y - _laserCloud->points[i].y,
                                              _laserCloud->points[i - k].z - _laserCloud->points[i].z);
        //归一化
        tmp.normalize();
        //存入front_norms
        front_norms.push_back(tmp);
        //左侧三个点到当前点的坐标差归一化再加权平均，表示左侧平面方向
        norm_front += (k/6.0)* tmp;
      }
      //存储右侧点到当前点归一化后的坐标差
      std::vector<Eigen::Vector3d> back_norms;
      for(int k = 1;k<4;k++){
        //这里代码是不是有问题？算的是左侧的？？？？
        float temp_depth = sqrt(_laserCloud->points[i - k].x * _laserCloud->points[i - k].x +
                        _laserCloud->points[i - k].y * _laserCloud->points[i - k].y +
                        _laserCloud->points[i - k].z * _laserCloud->points[i - k].z);

        if(temp_depth < 1){
          continue;
        }

        Eigen::Vector3d tmp = Eigen::Vector3d(_laserCloud->points[i + k].x - _laserCloud->points[i].x,
                                              _laserCloud->points[i + k].y - _laserCloud->points[i].y,
                                              _laserCloud->points[i + k].z - _laserCloud->points[i].z);
        tmp.normalize();
        back_norms.push_back(tmp);
        //右侧三个点到当前点的坐标差归一化再加权平均，表示右侧平面方向
        norm_back += (k/6.0)* tmp;
      }
      //求左右平面夹角余弦值
      double cc = fabs( norm_front.dot(norm_back) / (norm_front.norm()*norm_back.norm()) );
      //如果夹角大于15° 小于165°  有突起
      if(cc < 0.95){
        debugnum3++;
      }else{
        //否则就是第一种不可靠情况
        //所以101表示第一种不可靠情况
        CloudFeatureFlag[i] = 101;
      }

    }

  }//结束角点特征提取对点云的遍历

  pcl::PointCloud<PointType>::Ptr laserCloudCorner(new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType> cornerPointsSharp;

  std::vector<int> pointsLessSharp_ori;

  //记录平面特征点的数量
  int num_surf = 0;
  //记录角点特征点的数量
  int num_corner = 0;

  //push_back feature
  //排除前五个点和后五个点遍历点云
  for(int i = 5; i < cloudSize - 5; i ++){
    //计算当前点深度平方
    float dis = _laserCloud->points[i].x * _laserCloud->points[i].x
            + _laserCloud->points[i].y * _laserCloud->points[i].y
            + _laserCloud->points[i].z * _laserCloud->points[i].z;
    //距离过近不要
    if(dis < thLidarNearestDis*thLidarNearestDis) continue;
    
    //提取平面特征点 存索引，去除非法点后点云的索引
    if(CloudFeatureFlag[i] == 2){
      pointsLessFlat.push_back(i);
      num_surf++;
      continue;
    }
    //提取角点特征点 为什么要操作两次？？？和内存有关？？？
    //150 两个面交线上的点 100 第二种不可靠的点   说得通吗，可能都是根据实验自己随便改策略？？？
    if(CloudFeatureFlag[i] == 100 || CloudFeatureFlag[i] == 150){ //
      //存索引 去除非法点后点云的索引
      pointsLessSharp_ori.push_back(i);
      laserCloudCorner->push_back(_laserCloud->points[i]);
    }

  }
  //再遍历一次选中的角点特征点
  for(int i = 0; i < laserCloudCorner->points.size();i++){
      //存索引 去除非法点后点云的索引
      pointsLessSharp.push_back(pointsLessSharp_ori[i]);
      num_corner++;
  }

}


/**
*从msg里读取传感器发来的点云信息，把信息提取出来存入laserCloud
*msg: 传感器采集的点云消息，类型为livox_ros_driver::CustomMsgConstPtr
*laserCloud: 输入的原始点云，类型为pcl::PointCloud<PointType>::Ptr&，是一个指向PointCloud的智能指针引用
*laserConerFeature: 输出的角点特征点的点云，类型为pcl::PointCloud<PointType>::Ptr&，是一个指向PointCloud的智能指针引用
*laserSurfFeature: 输出的平面特征点的点云，类型为pcl::PointCloud<PointType>::Ptr&，是一个指向PointCloud的智能指针引用
*laserNonFeature: 输出的非特征点的点云，类型为pcl::PointCloud<PointType>::Ptr&，是一个指向PointCloud的智能指针引用
*msg_seg: 输出的点云分割结果，类型为sensor_msgs::PointCloud2，表示点云的格式和数据
*Used_Line: 设定的要使用的最大线数，类型为int，用于限制线数的范围，超过该范围的线将被忽略。如1，代表只使用第一条扫描线
*/
void LidarFeatureExtractor::FeatureExtract_with_segment(const livox_ros_driver::CustomMsgConstPtr &msg,
                                                        pcl::PointCloud<PointType>::Ptr& laserCloud,
                                                        pcl::PointCloud<PointType>::Ptr& laserConerFeature,
                                                        pcl::PointCloud<PointType>::Ptr& laserSurfFeature,
                                                        pcl::PointCloud<PointType>::Ptr& laserNonFeature,
                                                        sensor_msgs::PointCloud2 &msg_seg,
                                                        const int Used_Line){
  //清空点云
  laserCloud->clear();
  laserConerFeature->clear();
  laserSurfFeature->clear();
  laserCloud->clear();
  //预留点云空间
  laserCloud->reserve(15000*N_SCANS);
  // vlines 保存每条线的原始数据，每个元素都是点云指针，代表一条lidar线上的点云
  for(auto & ptr : vlines){
    ptr->clear();
  }
  //vcorner 每一个元素都是点云指针，代表一束线上的角点特征点
  for(auto & v : vcorner){
    v.clear();
  }
  //vsurf 每一个元素都是点云指针，代表一束线上的平面特征点
  for(auto & v : vsurf){
    v.clear();
  }

  //以下三个变量的定义是为了做点云分割设置的
  //获取输入点云点的数量
  int dnum = msg->points.size();

  //动态分配内存，存储点云点ID
  int *idtrans = (int*)calloc(dnum, sizeof(int));
  //动态分配内存，用于存储点云的(x, y, z, intensity)数据
  float *data=(float*)calloc(dnum*4,sizeof(float));
  //往data里存数据用
  int point_num = 0;

// 览沃 pointcloud2(PointXYZRTL) 点云格式，详细说明如下:
// float32 x               # X axis, unit:m
// float32 y               # Y axis, unit:m
// float32 z               # Z axis, unit:m
// float32 intensity       # the value is reflectivity, 0.0~255.0
// uint8 tag               # livox tag
// uint8 line              # laser number in lidar

// 览沃自定义数据包格式，详细说明如下 :
// Header header             # ROS standard message header
// uint64 timebase           # The time of first point
// uint32 point_num          # Total number of pointclouds
// uint8  lidar_id           # Lidar device id number
// uint8[3]  rsvd            # Reserved use
// CustomPoint[] points      # Pointcloud data

// 上述自定义数据包中的自定义点云（CustomＰoint）格式 :
// uint32 offset_time      # offset time relative to the base time
// float32 x               # X axis, unit:m
// float32 y               # Y axis, unit:m
// float32 z               # Z axis, unit:m
// uint8 reflectivity      # reflectivity, 0~255
// uint8 tag               # livox tag
// uint8 line              # laser number in lidar
  //获取消息中最后一个点的时间偏移，并转换成秒。
  double timeSpan = ros::Time().fromNSec(msg->points.back().offset_time).toSec();
  PointType point;
  for(const auto& p : msg->points){
    //获取当前点所在的线数，估计从0开始的
    int line_num = (int)p.line;
    //如果当前点所在的线数大于设定要使用的的最大线数(Used_Line)，则跳过该点，继续下一个点
    if(line_num > Used_Line-1) continue;
    //如果当前点的x坐标小于0.01m，说明该点无效，跳过该点，继续下一个点 为什么无效？？？距离过近可能会产生畸变
    if(p.x < 0.01) continue;
    //如果当前点的x、y或z坐标无限大或无限小，说明该点无效，跳过该点，继续下一个点
    if (!pcl_isfinite(p.x) ||
        !pcl_isfinite(p.y) ||
        !pcl_isfinite(p.z)) {
      continue;
    }
    point.x = p.x;
    point.y = p.y;
    point.z = p.z;
    point.intensity = p.reflectivity; //0~255
    //将当前点的offset_time转换成秒，并除以timeSpan，存储到point的normal_x字段中，将当前点所在的线数转换成浮点数，并存储到point的normal_y字段中
    point.normal_x = ros::Time().fromNSec(p.offset_time).toSec() /timeSpan;
    point.normal_y = _int_as_float(line_num);
    laserCloud->push_back(point);

    data[point_num*4+0] = point.x;
    data[point_num*4+1] = point.y;
    data[point_num*4+2] = point.z;
    data[point_num*4+3] = point.intensity;


    point_num++;
  }//结束遍历所有点，数据存入data

  //点云分割   
  // idtrans 点云ID    
  // data 点云的(x, y, z, intensity)数据
  // dnum  输入点云点的数量
  PCSeg pcseg;
  pcseg.DoSeg(idtrans,data,dnum);

  //size_t = unsigned long long 获取点云大小
  std::size_t cloud_num = laserCloud->size();
  //遍历点云中的点
  for(std::size_t i=0; i<cloud_num; ++i){
    //normal_y是所在雷达线数
    int line_idx = _float_as_int(laserCloud->points[i].normal_y);
    //第几个点（索引）存入normal_z
    laserCloud->points[i].normal_z = _int_as_float(i);
    //将点云放在对应雷达线数里 存入vlines 分线数存储点云
    vlines[line_idx]->push_back(laserCloud->points[i]);
  }

  //长度为N_SCANS的线程数组，为每个扫描线都创建一个线程
  std::thread threads[N_SCANS];
  for(int i=0; i<N_SCANS; ++i){
    //为每条扫描线创建一个线程，线程函数为LidarFeatureExtractor类中的detectFeaturePoint3函数，同时传入两个参数：vlines[i]和vcorner[i]的引用。
    /**
     * 使用可调用对象的成员函数构造函数：template <class F, class... Args, class M> explicit thread(F&& f, M&& m, Args&&... args);
     * 创建一个新线程，并将执行函数设置为std::mem_fn(f)(m, args...)，其中f是可调用对象的成员函数指针，m是可调用对象的指针或引用，args是函数的参数。
    */
   //调用当前类的成员函数，传入的参数是引用
    threads[i] = std::thread(&LidarFeatureExtractor::detectFeaturePoint3, this, std::ref(vlines[i]),std::ref(vcorner[i]));
  }

  for(int i=0; i<N_SCANS; ++i){
    //当调用join()函数时，主线程会被阻塞，直到第i个线程执行完成。如果线程还没有执行完毕，主线程将等待直到该线程结束。一旦线程执行完成，join()函数会返回，主线程继续执行后面的代码。
    threads[i].join();
  }

  int num_corner = 0;
  for(int i=0; i<N_SCANS; ++i){
    for(int j=0; j<vcorner[i].size(); ++j){
      //vcorner[i][j].normal_z 第i束激光的第j的角点特征点在点云中的索引 把laserCloud中对应的点的normal_z改为1 表示该点已被选为角点特征点
      laserCloud->points[_float_as_int(vlines[i]->points[vcorner[i][j]].normal_z)].normal_z = 1.0; 
      num_corner++;
    }
  }
  //只提取平面特征点和非特征点
  detectFeaturePoint2(laserCloud, laserSurfFeature, laserNonFeature);
  
  for(std::size_t i=0; i<cloud_num; ++i){
    //求点云中点的深度
    float dis = laserCloud->points[i].x * laserCloud->points[i].x
                + laserCloud->points[i].y * laserCloud->points[i].y
                + laserCloud->points[i].z * laserCloud->points[i].z;
    //idtrans是点云分类标签数组 normal_z=0代表是？？？
    if( idtrans[i] > 9 && dis < 50*50){
      laserCloud->points[i].normal_z = 0;
    }
  }

  pcl::PointCloud<PointType>::Ptr laserConerFeature_filter;
  laserConerFeature_filter.reset(new pcl::PointCloud<PointType>());
  laserConerFeature.reset(new pcl::PointCloud<PointType>());
  laserSurfFeature.reset(new pcl::PointCloud<PointType>());
  laserNonFeature.reset(new pcl::PointCloud<PointType>());
  for(const auto& p : laserCloud->points){
    //normal_z=1 代表是角点特征点
    if(std::fabs(p.normal_z - 1.0) < 1e-5)
      laserConerFeature->push_back(p);
  }

  //normal_z=2和3 代表是平面特征点和非特征点
  for(const auto& p : laserCloud->points){
    if(std::fabs(p.normal_z - 2.0) < 1e-5)
      laserSurfFeature->push_back(p);
    if(std::fabs(p.normal_z - 3.0) < 1e-5)
      laserNonFeature->push_back(p);
  }

}

void LidarFeatureExtractor::FeatureExtract_with_segment_hap(const livox_ros_driver::CustomMsgConstPtr &msg,
                                                            pcl::PointCloud<PointType>::Ptr& laserCloud,
                                                            pcl::PointCloud<PointType>::Ptr& laserConerFeature,
                                                            pcl::PointCloud<PointType>::Ptr& laserSurfFeature,
                                                            pcl::PointCloud<PointType>::Ptr& laserNonFeature,
                                                            sensor_msgs::PointCloud2 &msg_seg,
                                                            const int Used_Line){
  laserCloud->clear();
  laserConerFeature->clear();
  laserSurfFeature->clear();
  laserCloud->clear();
  laserCloud->reserve(15000*N_SCANS);
  //vlines保存每条扫面线上的点云，遍历每条扫描线
  for(auto & ptr : vlines){
    ptr->clear();
  }
  //清空角点特征点
  for(auto & v : vcorner){
    v.clear();
  }
  //清空平面特征点
  for(auto & v : vsurf){
    v.clear();
  }

  //获取输入点云点的数量
  int dnum = msg->points.size();
  //动态分配内存，存储点云点ID
  int *idtrans = (int*)calloc(dnum, sizeof(int));
  //动态分配内存，用于存储点云的(x, y, z, intensity)数据
  float *data=(float*)calloc(dnum*4,sizeof(float));
  //？？？？
  int point_num = 0;
  //获取消息中最后一个点的时间偏移，并转换成秒。
  double timeSpan = ros::Time().fromNSec(msg->points.back().offset_time).toSec();
  PointType point;
  for(const auto& p : msg->points){
    //获取当前点所在的线数
    int line_num = (int)p.line;
    //如果当前点所在的线数大于设定要使用的的最大线数(Used_Line)，则跳过该点，继续下一个点
    if(line_num > Used_Line-1) continue;
    if(p.x < 0.01) continue;
    if (!pcl_isfinite(p.x) ||
        !pcl_isfinite(p.y) ||
        !pcl_isfinite(p.z)) {
      continue;
    }
    point.x = p.x;
    point.y = p.y;
    point.z = p.z;
    point.intensity = p.reflectivity;
    point.normal_x = ros::Time().fromNSec(p.offset_time).toSec() /timeSpan;
    point.normal_y = _int_as_float(line_num);
    laserCloud->push_back(point);

    data[point_num*4+0] = point.x;
    data[point_num*4+1] = point.y;
    data[point_num*4+2] = point.z;
    data[point_num*4+3] = point.intensity;


    point_num++;
  }

  PCSeg pcseg;
  pcseg.DoSeg(idtrans,data,dnum);
//函数hap和不加hap的区别就是这段代码，hap缺少这段代码，少了多线程调用detectFeaturePoint3提取角点特征点的部分
  //获取点云大小
  std::size_t cloud_num = laserCloud->size();

  detectFeaturePoint2(laserCloud, laserSurfFeature, laserNonFeature);

  for(std::size_t i=0; i<cloud_num; ++i){
    float dis = laserCloud->points[i].x * laserCloud->points[i].x
                + laserCloud->points[i].y * laserCloud->points[i].y
                + laserCloud->points[i].z * laserCloud->points[i].z;
    if( idtrans[i] > 9 && dis < 50*50){
      laserCloud->points[i].normal_z = 0;
    }
  }

  pcl::PointCloud<PointType>::Ptr laserConerFeature_filter;
  laserConerFeature_filter.reset(new pcl::PointCloud<PointType>());
  laserConerFeature.reset(new pcl::PointCloud<PointType>());
  laserSurfFeature.reset(new pcl::PointCloud<PointType>());
  laserNonFeature.reset(new pcl::PointCloud<PointType>());
  for(const auto& p : laserCloud->points){
    if(std::fabs(p.normal_z - 1.0) < 1e-5)
      laserConerFeature->push_back(p);
  }

  for(const auto& p : laserCloud->points){
    if(std::fabs(p.normal_z - 2.0) < 1e-5)
      laserSurfFeature->push_back(p);
    if(std::fabs(p.normal_z - 3.0) < 1e-5)
      laserNonFeature->push_back(p);
  }

}

//只提取平面特征点和非特征点
//laserCloud传进来
void LidarFeatureExtractor::detectFeaturePoint2(pcl::PointCloud<PointType>::Ptr& cloud,
                                                pcl::PointCloud<PointType>::Ptr& pointsLessFlat,
                                                pcl::PointCloud<PointType>::Ptr& pointsNonFeature){
  //点云大小
  int cloudSize = cloud->points.size();
  //智能指针指向新的空间，并释放原来的空间
  pointsLessFlat.reset(new pcl::PointCloud<PointType>());
  pointsNonFeature.reset(new pcl::PointCloud<PointType>());
  //KdTree存储点云
  pcl::KdTreeFLANN<PointType>::Ptr KdTreeCloud;
  KdTreeCloud.reset(new pcl::KdTreeFLANN<PointType>);
  //设置用于构建Kd树的输入点云数据
  KdTreeCloud->setInputCloud(cloud);

  std::vector<int> _pointSearchInd;
  std::vector<float> _pointSearchSqDis;

  //KdTree K近邻搜索时的最近邻点数
  int num_near = 10;
  //点云遍历步长
  int stride = 1;
  //根据点的深度确定满足条件的平面特征点给定的范围
  int interval = 4;

  for(int i = 5; i < cloudSize - 5; i = i+stride) {
    //如果已经被选为角点特征点，跳过该点
    if(fabs(cloud->points[i].normal_z - 1.0) < 1e-5) {
      continue;
    }
    //判断特征值贡献度的一些阈值
    double thre1d = 0.5;
    double thre2d = 0.8;
    double thre3d = 0.5;
    double thre3d2 = 0.13;
    //求点的深度
    double disti = sqrt(cloud->points[i].x * cloud->points[i].x + 
                        cloud->points[i].y * cloud->points[i].y + 
                        cloud->points[i].z * cloud->points[i].z);

    if(disti < 30.0) {
      thre1d = 0.5;
      thre2d = 0.8;
      thre3d2 = 0.07;
      stride = 14;
      interval = 4;
    } else if(disti < 60.0) {
      stride = 10;
      interval = 3;
    } else {
      stride = 1;
      interval = 0;
    }

    if(disti > 100.0) {
      num_near = 6;
      //远点normal_z设为3
      cloud->points[i].normal_z = 3.0;
      //同时把该远点放入pointsNonFeature
      pointsNonFeature->points.push_back(cloud->points[i]);
      continue;
    } else if(disti > 60.0) {
      num_near = 8;
    } else {
      num_near = 10;
    }
    //K近邻搜索
    //_pointSearchInd：返回查询到的最近邻点的索引列表。该参数是一个输出参数，用于存储查询到的最近邻点在输入点云中的索引
    //_pointSearchSqDis：返回查询到的最近邻点与查询点之间的距离列表。该参数也是一个输出参数，用于存储查询到的最近邻点与查询点之间的距离
    KdTreeCloud->nearestKSearch(cloud->points[i], num_near, _pointSearchInd, _pointSearchSqDis);
    //第num_near近的点
    if (_pointSearchSqDis[num_near-1] > 5.0 && disti < 90.0) {
      continue;
    }

    Eigen::Matrix< double, 3, 3 > _matA1;
    _matA1.setZero();

    float cx = 0;
    float cy = 0;
    float cz = 0;
    //求当前点最近邻点的平均坐标
    for (int j = 0; j < num_near; j++) {
      cx += cloud->points[_pointSearchInd[j]].x;
      cy += cloud->points[_pointSearchInd[j]].y;
      cz += cloud->points[_pointSearchInd[j]].z;
    }
    cx /= num_near;
    cy /= num_near;
    cz /= num_near;

    float a11 = 0;
    float a12 = 0;
    float a13 = 0;
    float a22 = 0;
    float a23 = 0;
    float a33 = 0;
    for (int j = 0; j < num_near; j++) {
      float ax = cloud->points[_pointSearchInd[j]].x - cx;
      float ay = cloud->points[_pointSearchInd[j]].y - cy;
      float az = cloud->points[_pointSearchInd[j]].z - cz;

      a11 += ax * ax;
      a12 += ax * ay;
      a13 += ax * az;
      a22 += ay * ay;
      a23 += ay * az;
      a33 += az * az;
    }
    a11 /= num_near;
    a12 /= num_near;
    a13 /= num_near;
    a22 /= num_near;
    a23 /= num_near;
    a33 /= num_near;
    //最近邻点分布的协方差矩阵
    _matA1(0, 0) = a11;
    _matA1(0, 1) = a12;
    _matA1(0, 2) = a13;
    _matA1(1, 0) = a12;
    _matA1(1, 1) = a22;
    _matA1(1, 2) = a23;
    _matA1(2, 0) = a13;
    _matA1(2, 1) = a23;
    _matA1(2, 2) = a33;
    //计算特征值和特征向量
    /*这行代码创建了一个SelfAdjointEigenSolver对象saes，用于对3x3的实对称矩阵_matA1进行特征值和特征向量的计算。
    Eigen库提供了不同类型的特征值求解器，SelfAdjointEigenSolver适用于实对称矩阵，它保证了特征值是实数，
    特征向量是正交的。*/
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(_matA1);
    /*在这行代码中，saes.eigenvalues()返回计算得到的特征值的向量。saes.eigenvalues()[i]表示第i个特征值。这里我们
    使用[2]、[1]和[0]索引分别获取了特征值的第三个（最大特征值）、第二个和第一个（最小特征值）元素。变量a1d计算了特征
    值的第三个元素与第二个元素    之间的差，并除以特征值的第三个元素。这个计算是为了获得特征值中第三个元素对于特征值
    整体尺度的贡献程度。*/
    double a1d = (sqrt(saes.eigenvalues()[2]) - sqrt(saes.eigenvalues()[1])) / sqrt(saes.eigenvalues()[2]);
    /*这行代码类似于上一行，但是计算的是特征值的第二个元素与第一个元素之间的差，并除以特征值的第三个元素。这个计算是
    为了获得特征值中第二个元素对于特征值整体尺度的贡献程度*/
    double a2d = (sqrt(saes.eigenvalues()[1]) - sqrt(saes.eigenvalues()[0])) / sqrt(saes.eigenvalues()[2]);
    /*这行代码计算的是特征值的第一个元素除以特征值的第三个元素。这个计算是为了获得特征值中第一个元素对于特征值整体尺度的贡献程度。*/
    double a3d = sqrt(saes.eigenvalues()[0]) / sqrt(saes.eigenvalues()[2]);

    //如果满足条件，当前点前后interval个点（包括当前点）都算作平面特征点
    if(a2d > thre2d || (a3d < thre3d2 && a1d < thre1d)) {
      for(int k = 1; k < interval; k++) {
        //normal_z=2代表平面特征点
        cloud->points[i-k].normal_z = 2.0;
        pointsLessFlat->points.push_back(cloud->points[i-k]);
        cloud->points[i+k].normal_z = 2.0;
        pointsLessFlat->points.push_back(cloud->points[i+k]);
      }
      cloud->points[i].normal_z = 2.0;
      pointsLessFlat->points.push_back(cloud->points[i]);
    } else if(a3d > thre3d) {//满足条件算作非特征点
      for(int k = 1; k < interval; k++) {
        cloud->points[i-k].normal_z = 3.0;
        pointsNonFeature->points.push_back(cloud->points[i-k]);
        cloud->points[i+k].normal_z = 3.0;
        pointsNonFeature->points.push_back(cloud->points[i+k]);
      }
      cloud->points[i].normal_z = 3.0;
      pointsNonFeature->points.push_back(cloud->points[i]);
    }
  }  
}

//只提取角点特征点
void LidarFeatureExtractor::detectFeaturePoint3(pcl::PointCloud<PointType>::Ptr& cloud,
                                                std::vector<int>& pointsLessSharp){
  //存特征标志
  int CloudFeatureFlag[20000];
  //存曲率
  float cloudCurvature[20000];
  //存深度
  float cloudDepth[20000];
  //曲率从小到大的索引
  int cloudSortInd[20000];
  //存反射强度
  float cloudReflect[20000];
  //反射强度从小到大的索引
  int reflectSortInd[20000];
  //如果点与前后点的角度均小于15°，将该点标志置为1？？？？
  int cloudAngle[20000];
  //传进来的点云指针给引用
  pcl::PointCloud<PointType>::Ptr& laserCloudIn = cloud;
  //获取点云大小
  int cloudSize = laserCloudIn->points.size();

  PointType point;
  //存储去除非法点后的点云
  pcl::PointCloud<PointType>::Ptr _laserCloud(new pcl::PointCloud<PointType>());
  _laserCloud->reserve(cloudSize);

  for (int i = 0; i < cloudSize; i++) {
    point.x = laserCloudIn->points[i].x;
    point.y = laserCloudIn->points[i].y;
    point.z = laserCloudIn->points[i].z;
    point.normal_x = 1.0;
    point.intensity = laserCloudIn->points[i].intensity;

    if (!pcl_isfinite(point.x) ||
        !pcl_isfinite(point.y) ||
        !pcl_isfinite(point.z)) {
      continue;
    }

    _laserCloud->push_back(point);
    CloudFeatureFlag[i] = 0;
  }
  //获取去除非法点后的点云
  cloudSize = _laserCloud->size();

  int count_num = 1;
  bool left_surf_flag = false;
  bool right_surf_flag = false;

  //--------------------------------------------------- break points ---------------------------------------------
  for(int i = 5; i < cloudSize - 5; i ++){
    float diff_left[2];
    float diff_right[2];
    float depth = sqrt(_laserCloud->points[i].x * _laserCloud->points[i].x +
                       _laserCloud->points[i].y * _laserCloud->points[i].y +
                       _laserCloud->points[i].z * _laserCloud->points[i].z);

    for(int count = 1; count < 3; count++ ){
      float diffX1 = _laserCloud->points[i + count].x - _laserCloud->points[i].x;
      float diffY1 = _laserCloud->points[i + count].y - _laserCloud->points[i].y;
      float diffZ1 = _laserCloud->points[i + count].z - _laserCloud->points[i].z;
      diff_right[count - 1] = sqrt(diffX1 * diffX1 + diffY1 * diffY1 + diffZ1 * diffZ1);

      float diffX2 = _laserCloud->points[i - count].x - _laserCloud->points[i].x;
      float diffY2 = _laserCloud->points[i - count].y - _laserCloud->points[i].y;
      float diffZ2 = _laserCloud->points[i - count].z - _laserCloud->points[i].z;
      diff_left[count - 1] = sqrt(diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2);
    }

    float depth_right = sqrt(_laserCloud->points[i + 1].x * _laserCloud->points[i + 1].x +
                             _laserCloud->points[i + 1].y * _laserCloud->points[i + 1].y +
                             _laserCloud->points[i + 1].z * _laserCloud->points[i + 1].z);
    float depth_left = sqrt(_laserCloud->points[i - 1].x * _laserCloud->points[i - 1].x +
                            _laserCloud->points[i - 1].y * _laserCloud->points[i - 1].y +
                            _laserCloud->points[i - 1].z * _laserCloud->points[i - 1].z);

    
    if(fabs(diff_right[0] - diff_left[0]) > thBreakCornerDis){
      if(diff_right[0] > diff_left[0]){

        Eigen::Vector3d surf_vector = Eigen::Vector3d(_laserCloud->points[i - 1].x - _laserCloud->points[i].x,
                                                      _laserCloud->points[i - 1].y - _laserCloud->points[i].y,
                                                      _laserCloud->points[i - 1].z - _laserCloud->points[i].z);
        Eigen::Vector3d lidar_vector = Eigen::Vector3d(_laserCloud->points[i].x,
                                                       _laserCloud->points[i].y,
                                                       _laserCloud->points[i].z);
        double left_surf_dis = surf_vector.norm();
        //calculate the angle between the laser direction and the surface
        double cc = fabs( surf_vector.dot(lidar_vector) / (surf_vector.norm()*lidar_vector.norm()) );

        std::vector<PointType> left_list;
        double min_dis = 10000;
        double max_dis = 0;
        for(int j = 0; j < 4; j++){   //TODO: change the plane window size and add thin rod support
          left_list.push_back(_laserCloud->points[i - j]);
          Eigen::Vector3d temp_vector = Eigen::Vector3d(_laserCloud->points[i - j].x - _laserCloud->points[i - j - 1].x,
                                                        _laserCloud->points[i - j].y - _laserCloud->points[i - j - 1].y,
                                                        _laserCloud->points[i - j].z - _laserCloud->points[i - j - 1].z);

          if(j == 3) break;
          double temp_dis = temp_vector.norm();
          if(temp_dis < min_dis) min_dis = temp_dis;
          if(temp_dis > max_dis) max_dis = temp_dis;
        }
        // bool left_is_plane = plane_judge(left_list,0.3);

        if(cc < 0.93){//(max_dis < 2*min_dis) && left_surf_dis < 0.05 * depth  && left_is_plane &&
          if(depth_right > depth_left){
            CloudFeatureFlag[i] = 100;
          }
          else{
            if(depth_right == 0) CloudFeatureFlag[i] = 100;
          }
        }
      }
      else{

        Eigen::Vector3d surf_vector = Eigen::Vector3d(_laserCloud->points[i + 1].x - _laserCloud->points[i].x,
                                                      _laserCloud->points[i + 1].y - _laserCloud->points[i].y,
                                                      _laserCloud->points[i + 1].z - _laserCloud->points[i].z);
        Eigen::Vector3d lidar_vector = Eigen::Vector3d(_laserCloud->points[i].x,
                                                       _laserCloud->points[i].y,
                                                       _laserCloud->points[i].z);
        double right_surf_dis = surf_vector.norm();
        //calculate the angle between the laser direction and the surface
        double cc = fabs( surf_vector.dot(lidar_vector) / (surf_vector.norm()*lidar_vector.norm()) );

        std::vector<PointType> right_list;
        double min_dis = 10000;
        double max_dis = 0;
        for(int j = 0; j < 4; j++){ //TODO: change the plane window size and add thin rod support
          right_list.push_back(_laserCloud->points[i - j]);
          Eigen::Vector3d temp_vector = Eigen::Vector3d(_laserCloud->points[i + j].x - _laserCloud->points[i + j + 1].x,
                                                        _laserCloud->points[i + j].y - _laserCloud->points[i + j + 1].y,
                                                        _laserCloud->points[i + j].z - _laserCloud->points[i + j + 1].z);

          if(j == 3) break;
          double temp_dis = temp_vector.norm();
          if(temp_dis < min_dis) min_dis = temp_dis;
          if(temp_dis > max_dis) max_dis = temp_dis;
        }
        // bool right_is_plane = plane_judge(right_list,0.3);

        if(cc < 0.93){ //right_is_plane  && (max_dis < 2*min_dis) && right_surf_dis < 0.05 * depth &&

          if(depth_right < depth_left){
            CloudFeatureFlag[i] = 100;
          }
          else{
            if(depth_left == 0) CloudFeatureFlag[i] = 100;
          }
        }
      }
    }

    // break points select
    if(CloudFeatureFlag[i] == 100){
      std::vector<Eigen::Vector3d> front_norms;
      Eigen::Vector3d norm_front(0,0,0);
      Eigen::Vector3d norm_back(0,0,0);

      for(int k = 1;k<4;k++){

        float temp_depth = sqrt(_laserCloud->points[i - k].x * _laserCloud->points[i - k].x +
                        _laserCloud->points[i - k].y * _laserCloud->points[i - k].y +
                        _laserCloud->points[i - k].z * _laserCloud->points[i - k].z);

        if(temp_depth < 1){
          continue;
        }

        Eigen::Vector3d tmp = Eigen::Vector3d(_laserCloud->points[i - k].x - _laserCloud->points[i].x,
                                              _laserCloud->points[i - k].y - _laserCloud->points[i].y,
                                              _laserCloud->points[i - k].z - _laserCloud->points[i].z);
        tmp.normalize();
        front_norms.push_back(tmp);
        norm_front += (k/6.0)* tmp;
      }
      std::vector<Eigen::Vector3d> back_norms;
      for(int k = 1;k<4;k++){

        float temp_depth = sqrt(_laserCloud->points[i - k].x * _laserCloud->points[i - k].x +
                        _laserCloud->points[i - k].y * _laserCloud->points[i - k].y +
                        _laserCloud->points[i - k].z * _laserCloud->points[i - k].z);

        if(temp_depth < 1){
          continue;
        }

        Eigen::Vector3d tmp = Eigen::Vector3d(_laserCloud->points[i + k].x - _laserCloud->points[i].x,
                                              _laserCloud->points[i + k].y - _laserCloud->points[i].y,
                                              _laserCloud->points[i + k].z - _laserCloud->points[i].z);
        tmp.normalize();
        back_norms.push_back(tmp);
        norm_back += (k/6.0)* tmp;
      }
      double cc = fabs( norm_front.dot(norm_back) / (norm_front.norm()*norm_back.norm()) );
      if(cc < 0.93){
      }else{
        CloudFeatureFlag[i] = 101;
      }

    }

  }

  pcl::PointCloud<PointType>::Ptr laserCloudCorner(new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType> cornerPointsSharp;

  std::vector<int> pointsLessSharp_ori;

  int num_surf = 0;
  int num_corner = 0;

/**
 * 和detectFeaturePoint提取角点特征点的部分的区别在这里以下的部分
*/
  //遍历点云
  for(int i = 5; i < cloudSize - 5; i ++){
    Eigen::Vector3d left_pt = Eigen::Vector3d(_laserCloud->points[i - 1].x,
                                              _laserCloud->points[i - 1].y,
                                              _laserCloud->points[i - 1].z);
    Eigen::Vector3d right_pt = Eigen::Vector3d(_laserCloud->points[i + 1].x,
                                               _laserCloud->points[i + 1].y,
                                               _laserCloud->points[i + 1].z);

    Eigen::Vector3d cur_pt = Eigen::Vector3d(_laserCloud->points[i].x,
                                             _laserCloud->points[i].y,
                                             _laserCloud->points[i].z);
    //当前点深度
    float dis = _laserCloud->points[i].x * _laserCloud->points[i].x +
                _laserCloud->points[i].y * _laserCloud->points[i].y +
                _laserCloud->points[i].z * _laserCloud->points[i].z;
    //原点到左点的向量和原点到右点的向量之间夹角余弦值
    double clr = fabs(left_pt.dot(right_pt) / (left_pt.norm()*right_pt.norm()));
    //原点到左点的向量和原点到当前点的向量之间夹角余弦值
    double cl = fabs(left_pt.dot(cur_pt) / (left_pt.norm()*cur_pt.norm()));
    //原点到当前点的向量和原点到右点的向量之间夹角余弦值
    double cr = fabs(right_pt.dot(cur_pt) / (right_pt.norm()*cur_pt.norm()));
    //？？？？？？？？？？？？
    if(clr < 0.999){
      CloudFeatureFlag[i] = 200;
    }

    if(dis < thLidarNearestDis*thLidarNearestDis) continue;

    if(CloudFeatureFlag[i] == 100 || CloudFeatureFlag[i] == 200){ //
      pointsLessSharp_ori.push_back(i);
      laserCloudCorner->push_back(_laserCloud->points[i]);
    }
  }

  for(int i = 0; i < laserCloudCorner->points.size();i++){
      pointsLessSharp.push_back(pointsLessSharp_ori[i]);
      num_corner++;
  }

}


/**
 * @brief 检测CustomMsg的激光雷达特征点,不进行动态障碍物剔除，调用核心函数detectFeaturePoint()
 * 
 * @param[in] msg 需要检测原始CustomMsg
 * @param[out] laserCloud 将CustomMsg转换为pcl点云格式，进行了0.01近点剔除，其中点的normal法向量存储时间占比，线束信息，特征点分类，1角点，2平面点
 * @param[out] laserConerFeature 从laserCloud中提取的较少Coner特征
 * @param[out] laserSurfFeature 从laserCloud提取的较少平面特征
 * @param[in] Used_Line 线数
 * @param[in] lidar_type 雷达类型0-horizon  2-mid360
 */
void LidarFeatureExtractor::FeatureExtract(const livox_ros_driver::CustomMsgConstPtr &msg,
                                           pcl::PointCloud<PointType>::Ptr& laserCloud,
                                           pcl::PointCloud<PointType>::Ptr& laserConerFeature,
                                           pcl::PointCloud<PointType>::Ptr& laserSurfFeature,
                                           const int Used_Line,const int lidar_type){
  /*msg信息先存入laserCloud*/
  laserCloud->clear();
  laserConerFeature->clear();
  laserSurfFeature->clear();
  laserCloud->reserve(15000*N_SCANS);
  for(auto & ptr : vlines){
  ptr->clear();
  }
  for(auto & v : vcorner){
  v.clear();
  }
  for(auto & v : vsurf){
  v.clear();
  }
  double timeSpan = ros::Time().fromNSec(msg->points.back().offset_time).toSec();
  PointType point;
  for(const auto& p : msg->points){
  int line_num = (int)p.line;
  if(line_num > Used_Line-1) continue;
  //根据雷达类型 过滤过近的点
  if(lidar_type == 0||lidar_type == 1)
  {
      if(p.x < 0.01) continue;
  }
  else if(lidar_type == 2)
  {
      if(std::fabs(p.x) < 0.01) continue;
  }
//  if(p.x < 0.01) continue;
  point.x = p.x;
  point.y = p.y;
  point.z = p.z;
  point.intensity = p.reflectivity;
  //记录时间 当前点相对第一个点的偏移时间/最后一个点相对第一个点的偏移时间
  point.normal_x = ros::Time().fromNSec(p.offset_time).toSec() /timeSpan;
  //normal_y存所在雷达线数
  point.normal_y = _int_as_float(line_num);
  laserCloud->push_back(point);
  }
  //点云点数
  //点云中的点分到vlines中去
  std::size_t cloud_num = laserCloud->size();
  for(std::size_t i=0; i<cloud_num; ++i){
  int line_idx = _float_as_int(laserCloud->points[i].normal_y);
  //？？？？？
  laserCloud->points[i].normal_z = _int_as_float(i);
  vlines[line_idx]->push_back(laserCloud->points[i]);
  //？？？？？
  laserCloud->points[i].normal_z = 0;
  }
  //多线程调用detectFeaturePoint
  std::thread threads[N_SCANS];
  for(int i=0; i<N_SCANS; ++i){
  threads[i] = std::thread(&LidarFeatureExtractor::detectFeaturePoint, this, std::ref(vlines[i]),
                     std::ref(vcorner[i]), std::ref(vsurf[i]));
  }
  for(int i=0; i<N_SCANS; ++i){
  threads[i].join();
  }
  for(int i=0; i<N_SCANS; ++i){
  for(int j=0; j<vcorner[i].size(); ++j){
  //normal_z=1代表角点特征点
  laserCloud->points[_float_as_int(vlines[i]->points[vcorner[i][j]].normal_z)].normal_z = 1.0;
  }
  for(int j=0; j<vsurf[i].size(); ++j){
  //normal_z=2代表平面特征点
  laserCloud->points[_float_as_int(vlines[i]->points[vsurf[i][j]].normal_z)].normal_z = 2.0;
  }
  }

  for(const auto& p : laserCloud->points){
  if(std::fabs(p.normal_z - 1.0) < 1e-5)
  laserConerFeature->push_back(p);
  }
  for(const auto& p : laserCloud->points){
  if(std::fabs(p.normal_z - 2.0) < 1e-5)
  laserSurfFeature->push_back(p);
  }
}

void LidarFeatureExtractor::FeatureExtract_hap(const livox_ros_driver::CustomMsgConstPtr &msg,
                                               pcl::PointCloud<PointType>::Ptr& laserCloud,
                                               pcl::PointCloud<PointType>::Ptr& laserConerFeature,
                                               pcl::PointCloud<PointType>::Ptr& laserSurfFeature,
                                               pcl::PointCloud<PointType>::Ptr& laserNonFeature,
                                               const int Used_Line){
  laserCloud->clear();
  laserConerFeature->clear();
  laserSurfFeature->clear();
  laserCloud->clear();
  laserCloud->reserve(15000*N_SCANS);
  for(auto & ptr : vlines){
    ptr->clear();
  }
  for(auto & v : vcorner){
    v.clear();
  }
  for(auto & v : vsurf){
    v.clear();
  }

  int dnum = msg->points.size();

  double timeSpan = ros::Time().fromNSec(msg->points.back().offset_time).toSec();
  PointType point;
  for(const auto& p : msg->points){

    int line_num = (int)p.line;
    if(line_num > Used_Line-1) continue;
    if(p.x < 0.01) continue;
    if (!pcl_isfinite(p.x) ||
        !pcl_isfinite(p.y) ||
        !pcl_isfinite(p.z)) {
      continue;
    }
    point.x = p.x;
    point.y = p.y;
    point.z = p.z;
    point.intensity = p.reflectivity;
    point.normal_x = ros::Time().fromNSec(p.offset_time).toSec() /timeSpan;
    point.normal_y = _int_as_float(line_num);
    laserCloud->push_back(point);
  }
  //和FeatureExtract的区别之处
  detectFeaturePoint2(laserCloud, laserSurfFeature, laserNonFeature);

  pcl::PointCloud<PointType>::Ptr laserConerFeature_filter;
  laserConerFeature_filter.reset(new pcl::PointCloud<PointType>());
  laserConerFeature.reset(new pcl::PointCloud<PointType>());
  laserSurfFeature.reset(new pcl::PointCloud<PointType>());
  laserNonFeature.reset(new pcl::PointCloud<PointType>());
  for(const auto& p : laserCloud->points){
    if(std::fabs(p.normal_z - 1.0) < 1e-5)
      laserConerFeature->push_back(p);
  }

  for(const auto& p : laserCloud->points){
    if(std::fabs(p.normal_z - 2.0) < 1e-5)
      laserSurfFeature->push_back(p);
    if(std::fabs(p.normal_z - 3.0) < 1e-5)
      laserNonFeature->push_back(p);
  }
}

//全部线束都取 调用detectFeaturePoint
void LidarFeatureExtractor::FeatureExtract_Mid(pcl::PointCloud<pcl::PointXYZINormal>::Ptr &msg,
                                           pcl::PointCloud<PointType>::Ptr& laserConerFeature,
                                           pcl::PointCloud<PointType>::Ptr& laserSurfFeature){
    laserConerFeature->clear();
    laserSurfFeature->clear();
    for(auto & ptr : vlines){
        ptr->clear();
    }
    for(auto & v : vcorner){
        v.clear();
    }
    for(auto & v : vsurf){
        v.clear();
    }
    int cloud_num= msg->points.size();
    for(int i=0; i<cloud_num; ++i){
        int line_idx = std::round(msg->points[i].normal_y);
        msg->points[i].normal_z = _int_as_float(i);

        vlines[line_idx]->push_back(msg->points[i]);

        msg->points[i].normal_z = 0;
    }
    std::thread threads[N_SCANS];
    for(int i=0; i<N_SCANS; ++i){
        threads[i] = std::thread(&LidarFeatureExtractor::detectFeaturePoint, this, std::ref(vlines[i]),
                                 std::ref(vcorner[i]), std::ref(vsurf[i]));
    }
    for(int i=0; i<N_SCANS; ++i){
        threads[i].join();
    }
    for(int i=0; i<N_SCANS; ++i){
        for(int j=0; j<vcorner[i].size(); ++j){
            msg->points[_float_as_int(vlines[i]->points[vcorner[i][j]].normal_z)].normal_z = 1.0;
        }
        for(int j=0; j<vsurf[i].size(); ++j){
            msg->points[_float_as_int(vlines[i]->points[vsurf[i][j]].normal_z)].normal_z = 2.0;
        }
    }
    for(const auto& p : msg->points){
        if(std::fabs(p.normal_z - 1.0) < 1e-5)
            laserConerFeature->push_back(p);
    }
    for(const auto& p : msg->points){
        if(std::fabs(p.normal_z - 2.0) < 1e-5)
            laserSurfFeature->push_back(p);
    }
}