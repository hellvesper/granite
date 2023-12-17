#pragma once
#include <vector>
#include "types.h"
#include <numeric>
#include "sophus/se3.hpp"
#include "sophus/sim3.hpp"
#include "Eigen/Eigen"

#include <iostream>
#include <fstream>

namespace pose_graph_solver {
int findNearestFrame(const std::vector<granite::GPSconstraint>& constraints,
                     int64_t queryFrameTimestamp) {
  int64_t smallest_distance = std::numeric_limits<int64_t>::max();
  int idx = 0;
  for (size_t i = 0; i < constraints.size(); ++i)
  {
    auto constraint = constraints[i];
    int64_t current_distance =
        std::abs(constraint.timestamp - queryFrameTimestamp);

    if (current_distance < smallest_distance)
    {
      smallest_distance = current_distance;
      idx = i;
    }
  }

  double distanceInSeconds = (double)smallest_distance / 1e9;

  if (distanceInSeconds < 0.2) {
    return idx;
  } else {
    return 1'000'000;
  }
}
void findAlignment(Eigen::aligned_map<int64_t, Sophus::SE3d>& allFrames,
                   const std::vector<granite::GPSconstraint>& constraints,
                   granite::alignmentSe3& alignemt)
{
  if (allFrames.size() < 60)
  {
    return;
  }
 std::vector<Eigen::Vector3d> modelTranslations;
 std::vector<Eigen::Vector3d> dataTranslations;

 for (auto& sample : allFrames)
 {
   int alignedFrame = findNearestFrame(constraints, sample.first);

   if (alignedFrame > 10000)
   {
     return;
   }

   dataTranslations.push_back(constraints[alignedFrame].p);
   modelTranslations.push_back(sample.second.translation());
 }

 // constraints
 Eigen::Vector3d modelMean = Eigen::Vector3d::Zero();
 // frames
 Eigen::Vector3d dataMean = Eigen::Vector3d::Zero();

 for (size_t i = 0; i < modelTranslations.size(); ++i)
 {
   modelMean += modelTranslations[i];
   dataMean += dataTranslations[i];
 }

 modelMean /= modelTranslations.size();
 dataMean /= modelTranslations.size();


 for (size_t i = 0; i < modelTranslations.size(); ++i)
 {
   modelTranslations[i] -= modelMean;
   dataTranslations[i] -= dataMean;
 }

 Eigen::Matrix3d w = Eigen::Matrix3d::Zero();

 for (size_t i = 0; i < modelTranslations.size(); ++i)
 {
   w += dataTranslations[i] * modelTranslations[i].transpose();
 }

 Eigen::JacobiSVD<Eigen::MatrixXd> svd(w, Eigen::ComputeFullU | Eigen::ComputeFullV);

 Eigen::Matrix3d u = svd.matrixU();
 Eigen::Matrix3d v = svd.matrixV().transpose();
 Eigen::Matrix3d s = Eigen::Matrix3d::Identity();

 if (u.determinant() * v.determinant() < 0)
 {
   s(2, 2) = -1;
 }

 Eigen::Matrix3d rot = u * s * v;

 Eigen::Vector3d trans = dataMean - rot * modelMean;

 Eigen::Affine3d A;
 A.linear() = rot;
 A.translation() = trans;

 auto sv = svd.singularValues();
 alignemt.aligment.rotationMatrix() = A.matrix().block<3, 3>(0, 0);
 alignemt.aligment.translation() = A.matrix().block<3, 1>(0, 3);

 alignemt.singValuePercent = sv.y() / sv.x();
}

void findAlignment2(std::vector<granite::GPSconstraint>& constraints)
{
  auto algn = Sophus::SE3d(constraints[0].q, constraints[0].p).inverse();

  // 'timestamp tx ty tz qx qy qz qw'

  // std::ofstream myfile;
  // myfile.open ("/home/artem/example.txt");

  //T_global_pc
  // T_pc_global * T_global_pc
  for (auto& fr : constraints)
  {
    auto q = Sophus::SE3d(fr.q, fr.p);

    auto newPose = algn * q;

    //myfile << fr.timestamp << " " << newPose.translation().x() << " " << newPose.translation().y() << " " << newPose.translation().z() << " ";
    //myfile << newPose.unit_quaternion().x() << " " << newPose.so3().unit_quaternion().y() << " " << newPose.unit_quaternion().z() << " ";
    //myfile << newPose.unit_quaternion().w() << "\n";

    //std::cout << "new pose: \n" << newPose.matrix3x4() << std::endl;

    fr.world_pose = newPose;
    //fr.orig = q;

    //std::cout << "new pose2: \n" << fr.world_pose.matrix3x4() << std::endl;
  }

  //myfile.close();
}

}