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
/**
 * This function gets vector of GPS constraints, image timestamp and returns
 * closest GPS pose.
 * @param constraints The vector of constraints.
 * @param queryFrameTimestamp Image timetamp.
 * @return The closes index in vector of GPS poses.
 */
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

/**
 * Helper function to conver eigen pose/position to sophus pose.
 * @param constraints Vector of GPS constraints.
 */
void converQuatPositionToSophus(std::vector<granite::GPSconstraint>& constraints)
{
  for (auto& fr : constraints)
  {
    auto q = Sophus::SE3d(fr.q, fr.p);
    auto newPose = q;
    fr.world_pose = newPose;
  }
}

/**
 * This function realigns global constraints to the local
 * coordinate system by extracting the relative pose and
 * scaling translation part between them by value
 * [visual translation / gps translation].
 * @param constraints Vector of GPS constraints.
 * @param frame_poses Actual visual poses of SLAM
 */
void realign(std::vector<granite::GPSconstraint>& constraints,
             Eigen::aligned_map<int64_t, granite::PoseStateWithLin>& frame_poses)
{
  std::vector<int64_t> indexes;

  for (auto& frame : frame_poses)
  {
    indexes.push_back(frame.first);
  }

  std::sort(indexes.begin(), indexes.end());

  double displacement_vis =
      (frame_poses[indexes[0]].getPose().inverse() * frame_poses[indexes[indexes.size() - 1]].getPose())
      .translation().norm();

  auto idx1 = findNearestFrame(constraints, indexes[0]);
  auto idx2 = findNearestFrame(constraints, indexes[indexes.size() - 1]);

  double displacement_gps =
      (constraints[idx1].world_pose.inverse() * constraints[idx2].world_pose)
          .translation().norm();

  double scale = displacement_vis / displacement_gps;

  std::vector<Sophus::SE3d> relative_poses;
  for (size_t i = 1; i < constraints.size(); ++i)
  {
    auto pose_prev = constraints[i - 1].world_pose;
    auto pose_curr = constraints[i].world_pose;

    auto rel = pose_prev.inverse() * pose_curr;
    rel.translation() *= scale;

    relative_poses.push_back(rel);
  }

  constraints[0].realigned_pose = Sophus::SE3d();
  constraints[0].realigned = true;

  for (size_t i = 1; i < constraints.size(); ++i)
  {
    constraints[i].realigned_pose = constraints[i - 1].realigned_pose * relative_poses[i - 1];
    constraints[i].realigned = true;
  }
}

}