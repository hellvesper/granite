#pragma once

#include "types.h"
#include "relative_pose_error.h"
#include "ceres/ceres.h"
#include "aligner.h"

namespace pose_graph_solver
{
// Constructs the nonlinear least squares optimization problem from the pose
// graph constraints.
/**
 * This function builds a pose graph problem between visual poses and GPS poses.
 *
 * @param constraints A vector of constraints between GPS pose and visual pose.
 * @param constraintsCeres_poses A vector that contains constraints between visual poses (actually measured from
 * the visual solution).
 * @param constraintsCeres_gps_rel A vector that contains constraints between corresponding GPS poses
 * @param poses_frames A vector of poses that will be optimised.
 * @param poses_gps A vector of GPS poses that will be used as fixed poses in graph.
 * @param problem A ceres problem instance.
 * @param unfixed This represents the unfixed fram index.
 */
void BuildOptimizationProblem(const ceres::examples::VectorOfConstraints& constraints,
                              const ceres::examples::VectorOfConstraints& constraintsCeres_poses,
                              const ceres::examples::VectorOfConstraints& constraintsCeres_gps_rel,
                              ceres::examples::MapOfPoses* poses_frames,
                              ceres::examples::MapOfPoses* poses_gps,
                              ceres::Problem* problem,
                              int64_t unfixed)
{
  ceres::LossFunction* loss_function = new ceres::CauchyLoss(2.7955321);
  ceres::Manifold* quaternion_manifold = new ceres::EigenQuaternionManifold;

  // This block adds residuals  between absolute visual and aligned GPS poses.
  for (const auto& constraint : constraints) {
    auto pose_begin_iter = poses_frames->find(constraint.timestamp);
    auto pose_end_iter = poses_gps->find(constraint.timestamp);

    const Eigen::Matrix<double, 6, 6> sqrt_information =
        Eigen::Matrix<double, 6, 6>::Identity() * 0.0001;
    // Ceres will take ownership of the pointer.
    ceres::CostFunction* cost_function =
        PoseGraph3dErrorTerm::Create(constraint.t_be, sqrt_information);

    problem->AddResidualBlock(cost_function,
                              loss_function,
                              pose_begin_iter->second.p.data(),
                              pose_begin_iter->second.q.coeffs().data(),
                              pose_end_iter->second.p.data(),
                              pose_end_iter->second.q.coeffs().data());

    problem->SetManifold(pose_begin_iter->second.q.coeffs().data(),
                         quaternion_manifold);
    problem->SetManifold(pose_end_iter->second.q.coeffs().data(),
                         quaternion_manifold);

    problem->SetParameterBlockConstant(pose_end_iter->second.p.data());
    problem->SetParameterBlockConstant(pose_end_iter->second.q.coeffs().data());
  }

  // This block adds residual between absolute visual poses using their visual relative measurement between them.
  for (const auto& constraint : constraintsCeres_poses) {
    auto pose_begin_iter = poses_frames->find(constraint.prev);
    auto pose_end_iter = poses_frames->find(constraint.curr);

    Eigen::Matrix<double, 6, 6> sqrt_information =
        Eigen::Matrix<double, 6, 6>::Identity();
    // Ceres will take ownership of the pointer.
    ceres::CostFunction* cost_function =
        PoseGraph3dErrorTerm::Create(constraint.t_be, sqrt_information);

    problem->AddResidualBlock(cost_function,
                              loss_function,
                              pose_begin_iter->second.p.data(),
                              pose_begin_iter->second.q.coeffs().data(),
                              pose_end_iter->second.p.data(),
                              pose_end_iter->second.q.coeffs().data());

    problem->SetManifold(pose_begin_iter->second.q.coeffs().data(),
                         quaternion_manifold);
    problem->SetManifold(pose_end_iter->second.q.coeffs().data(),
                         quaternion_manifold);
  }

  // This block adds residual between absolute visual poses using their GPS relative measurement between them.
  for (const auto& constraint : constraintsCeres_gps_rel) {
    auto pose_begin_iter = poses_frames->find(constraint.prev);
    auto pose_end_iter = poses_frames->find(constraint.curr);

    const Eigen::Matrix<double, 6, 6> sqrt_information =
        Eigen::Matrix<double, 6, 6>::Identity() * 0.01;
    // Ceres will take ownership of the pointer.
    ceres::CostFunction* cost_function =
        PoseGraph3dErrorTerm::Create(constraint.t_be, sqrt_information);

    problem->AddResidualBlock(cost_function,
                              loss_function,
                              pose_begin_iter->second.p.data(),
                              pose_begin_iter->second.q.coeffs().data(),
                              pose_end_iter->second.p.data(),
                              pose_end_iter->second.q.coeffs().data());

    problem->SetManifold(pose_begin_iter->second.q.coeffs().data(),
                         quaternion_manifold);
    problem->SetManifold(pose_end_iter->second.q.coeffs().data(),
                         quaternion_manifold);
  }

  /// fix poses (only the last frame will be unfixed)
  for (auto& pp : *poses_frames)
  {
    if (pp.first != unfixed)
    {
      problem->SetParameterBlockConstant(pp.second.p.data());
      problem->SetParameterBlockConstant(pp.second.q.coeffs().data());
    }
  }
}

// Returns true if the solve was successful.
bool SolveOptimizationProblem(ceres::Problem* problem)
{
  ceres::Solver::Options options;
  options.max_num_iterations = 200;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

  ceres::Solver::Summary summary;
  ceres::Solve(options, problem, &summary);

  return summary.IsSolutionUsable();
}

}
