/**
MIT License

This file is part of the Granite project which is based on Basalt.
https://github.com/DLR-RM/granite

Copyright (c) Martin Wudenka, Deutsches Zentrum für Luft- und Raumfahrt

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

/**
Original license of Basalt:
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <granite/utils/assert.h>
#include <granite/utils/exceptions.h>
#include <granite/vi_estimator/keypoint_vo.h>
#include <granite/vi_estimator/mono_map_initialization.h>

#include <granite/optimization/accumulator.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <ceres/ceres.h>
#include <granite/vi_estimator/relative_pose_error.h>
#include <fstream>
#include <chrono>
#include <granite/vi_estimator/aligner.h>
#include <granite/vi_estimator/pose_graph_solver.h>

namespace granite {

KeypointVoEstimator::KeypointVoEstimator(
    const granite::Calibration<double>& calib, const VioConfig& config)
    : BundleAdjustmentBase(calib, config),
      take_kf(true),
      frames_after_kf(0),
      lambda(config.vio_lm_lambda_min),
      min_lambda(config.vio_lm_lambda_min),
      max_lambda(config.vio_lm_lambda_max),
      lambda_vee(2) {
  // Setup marginalization
  marg_H.setZero(se3_SIZE, se3_SIZE);
  marg_b.setZero(se3_SIZE);

  // prior on pose
  marg_H.diagonal().setConstant(config.vio_init_pose_weight);

  std::cout << "marg_H\n" << marg_H << std::endl;

  max_states = config.vio_max_states;
  max_kfs = config.vio_max_kfs;
}

void KeypointVoEstimator::reset() {
  BundleAdjustmentBase::reset();
  VioEstimatorBase::reset();

  // Setup marginalization
  marg_H.setZero(se3_SIZE, se3_SIZE);
  marg_b.setZero(se3_SIZE);

  // prior on pose
  marg_H.diagonal().setConstant(config.vio_init_pose_weight);

  std::cout << "marg_H\n" << marg_H << std::endl;

  negative_entropy_last_frame = 0;
  average_negative_entropy_last_frame = 0;
  map_initialized = false;

  prev_opt_flow_res.clear();
  frames_after_kf = 0;
  take_kf = true;
  this_unconnected_obs.clear();
  this_untriangulated_obs.clear();
  last_unconnected_obs.clear();
  last_untriangulated_obs.clear();

  tracking_state = TrackingState::UNINITIALIZED;
}

void KeypointVoEstimator::init_first_pose(const FrameId t_ns,
                                          const Sophus::SE3d& T_w_i) {
  tracking_state = TrackingState::TRACKING;

  this_state_t_ns = t_ns;
  frame_poses[t_ns] = PoseStateWithLin(t_ns, T_w_i, true);

  marg_order.abs_order_map.clear();
  marg_order.abs_order_map[t_ns] = std::make_pair(0, se3_SIZE);
  marg_order.total_size = se3_SIZE;
  marg_order.items = 1;

  std::cout << "Setting up filter: t_ns " << t_ns << std::endl;
  std::cout << "T_w_i\n" << T_w_i.matrix() << std::endl;

  T_w_i_init = T_w_i;
}

void KeypointVoEstimator::initialize(FrameId t_ns, const Sophus::SE3d& T_w_i,
                                     const Eigen::Vector3d& vel_w_i,
                                     const Eigen::Vector3d& bg,
                                     const Eigen::Vector3d& ba) {
  UNUSED(vel_w_i);
  UNUSED(bg);
  UNUSED(ba);

  init_first_pose(t_ns, T_w_i);
  initialize(bg, ba);
}

void KeypointVoEstimator::pushPoseConstraints(std::vector<granite::GPSconstraint>& poseConstraints)
{
  pose_graph_solver::converQuatPositionToSophus(poseConstraints);
  this->poseConstraints = poseConstraints;
}

/////////////////////////////////////////////////////////////////////////////////////////////
void KeypointVoEstimator::initialize(const Eigen::Vector3d& bg,
                                     const Eigen::Vector3d& ba) {
  auto proc_func = [&, bg, ba] {
    OpticalFlowResult::Ptr curr_frame;

    if (!vision_data_queue) {
      std::cerr << "Vision data queue is not present" << std::endl;
      return;
    }

    while (!should_quit)
    {
      try
      {
        vision_data_queue->pop(curr_frame);
      } catch (const tbb::user_abort&)
      {
        curr_frame = nullptr;
      };

      if (config.vio_enforce_realtime) {
        // drop current frame if another frame is already in the queue.
        while (!vision_data_queue->empty()) {
          try {
            vision_data_queue->pop(curr_frame);
          } catch (const tbb::user_abort&) {
            curr_frame = nullptr;
          };
        }
      }

      if (!curr_frame.get())
      {
        break;
      }

      // Correct camera time offset
      // curr_frame->t_ns += calib.cam_time_offset_ns;

      if (imu_data_queue)
      {
        while (!imu_data_queue->empty())
        {
          ImuData::Ptr d;
          imu_data_queue->pop(d);
        }
      }

      this_state_t_ns = curr_frame->t_ns;

      if (tracking_state == TrackingState::UNINITIALIZED)
      {
        std::cout << "TrackingState::UNINITIALIZED" << std::endl;
        init_first_pose(curr_frame->t_ns, Sophus::SE3d());
      }
      else if (prev_state_t_ns >= 0)
      {
        // init new state with pose of last state
        // const PoseStateWithLin& prev_state = frame_poses.at(prev_state_t_ns);

        // I want to use the average of the estimated visual position and the GPS position because
        // the first alignment is very rough and can be quite far from the estimated visual
        // pose and can blow up the solution, so make it smoother by using the average
        // of the visual and GPS.
        Eigen::Matrix<double, 3, 1> mean_position;

        auto pose_constraint_idx = pose_graph_solver::findNearestFrame(poseConstraints, prev_state_t_ns);
        if (poseConstraints[0].realigned)
        {
          mean_position = (poseConstraints[pose_constraint_idx].realigned_pose.translation() +
                  frame_poses.at(prev_state_t_ns).getPose().translation()) / 2.0;
        }

        auto hintPose = poseConstraints[pose_constraint_idx].realigned ? poseConstraints[pose_constraint_idx].realigned_pose :
                                                    frame_poses.at(prev_state_t_ns).getPose();
        if (poseConstraints[0].realigned)
        {
          hintPose.translation() = mean_position;
        }

        auto curr_state_p = poseConstraints[pose_constraint_idx].realigned ? hintPose :
                                                         frame_poses.at(prev_state_t_ns).getPose();

        PoseStateWithLin curr_state(this_state_t_ns, curr_state_p);

        /*const PoseStateWithLin& prev_state = frame_poses.at(prev_state_t_ns);
        PoseStateWithLin curr_state(this_state_t_ns, prev_state.getPose());*/


        frame_poses[this_state_t_ns] = curr_state;
      }


      if (!processFrame(curr_frame))
      {
        break;
      }


    }

    if (out_vis_queue) out_vis_queue->push(nullptr);
    if (out_marg_queue) out_marg_queue->push(nullptr);
    if (out_state_queue) out_state_queue->push(nullptr);

    quit();

    finished = true;

    std::cout << "Finished VIOFilter " << std::endl;
  };

  processing_thread.reset(new std::thread(proc_func));
}

/**
 * This function builds and optimises graph of poses using GPS data and
 * visual constraints between them. We want to improve our visual poses
 * by using relative GPS pose between them. After optimisation of the graph this functiuon
 * will update the
 * current pose (largest timestamp).
 * @param frame_poses The vector of actual SLAM frames poses.
 * @param constraints A vector of GPS data.
 */
static void optimise_graph(Eigen::aligned_map<int64_t, PoseStateWithLin>& frame_poses,
                           std::vector<granite::GPSconstraint>& constraints)
{
  // this vector will contain constraint between aligned GPS pose and visual pose
  ceres::examples::VectorOfConstraints constraintsCeres;
  // this vector will contain constraint between visual pose (relative pose between them itself)
  ceres::examples::VectorOfConstraints constraintsCeres_poses;
  // this vector will contain corresponding relative GPS pose between two visual frames.
  ceres::examples::VectorOfConstraints constraintsCeres_gps_rel;
  // vector of visual poses
  ceres::examples::MapOfPoses poses_frames;
  // vector of aligned GPS poses
  ceres::examples::MapOfPoses poses_gps;

  std::vector<int64_t> indexes;
  for (auto& pair : frame_poses)
  {
    indexes.push_back(pair.first);
    // add visual poses itself
    auto quat = Eigen::Quaterniond(pair.second.getPose().unit_quaternion().w(),
                                   pair.second.getPose().unit_quaternion().x(),
                                   pair.second.getPose().unit_quaternion().y(),
                                   pair.second.getPose().unit_quaternion().z());
    auto t = pair.second.getPose().translation();

    ceres::examples::Pose3d newPose;
    newPose.q = quat;
    newPose.p = t;

    poses_frames[pair.first] = newPose;

    ///  constraint
    // add constraints between visual <-> gps poses (should be identity)
    auto sophusConstraintRel = Sophus::SE3d();
    sophusConstraintRel.setRotationMatrix(Sophus::Matrix3d::Identity());
    auto quat_rel = Eigen::Quaterniond(sophusConstraintRel.unit_quaternion().w(),
                                       sophusConstraintRel.unit_quaternion().x(),
                                       sophusConstraintRel.unit_quaternion().y(),
                                       sophusConstraintRel.unit_quaternion().z());
    auto t_rel = sophusConstraintRel.translation();

    ceres::examples::Pose3d newPoseConstraint;
    newPoseConstraint.q = quat_rel;
    newPoseConstraint.p = t_rel;

    ceres::examples::Constraint3d constraint;
    constraint.timestamp = pair.first;
    constraint.t_be = newPoseConstraint;
    constraintsCeres.push_back(constraint);

    ///  gps
    // add relative gps constraint
    auto idx_gps = pose_graph_solver::findNearestFrame(constraints, pair.first);
    auto sophusGps = constraints[idx_gps].realigned_pose;

    auto quat_gps = Eigen::Quaterniond(sophusGps.unit_quaternion().w(),
                                       sophusGps.unit_quaternion().x(),
                                       sophusGps.unit_quaternion().y(),
                                       sophusGps.unit_quaternion().z());
    auto t_gps = sophusGps.translation();

    ceres::examples::Pose3d newPoseGps;
    newPoseGps.q = quat_gps;
    newPoseGps.p = t_gps;
    poses_gps[pair.first] = newPoseGps;
  }

  std::sort(indexes.begin(), indexes.end());

  // add relative visual pose constraint
  if (indexes.size() > 1)
  {
    for (size_t i = 1; i < indexes.size(); ++i)
    {
      int64_t timestamp_prev = indexes[i - 1];
      int64_t timestamp_curr = indexes[i];

      auto posePrev = frame_poses[timestamp_prev].getPose();
      auto poseCurr = frame_poses[timestamp_curr].getPose();
      auto rel = posePrev.inverse() * poseCurr;

      auto quat_rel = Eigen::Quaterniond(rel.unit_quaternion().w(),
                                         rel.unit_quaternion().x(),
                                         rel.unit_quaternion().y(),
                                         rel.unit_quaternion().z());
      auto t_rel = rel.translation();

      ceres::examples::Pose3d newPoseConstraint;
      newPoseConstraint.q = quat_rel;
      newPoseConstraint.p = t_rel;

      ceres::examples::Constraint3d constraint;
      constraint.prev = timestamp_prev;
      constraint.curr = timestamp_curr;
      constraint.t_be = newPoseConstraint;
      constraintsCeres_poses.push_back(constraint);
    }
  }

  // add relative GPS pose constraint
  if (indexes.size() > 1)
  {
    for (size_t i = 1; i < indexes.size(); ++i)
    {
      int64_t timestamp_prev = indexes[i - 1];
      int64_t timestamp_curr = indexes[i];

      auto poseConstrainIdxPrev = pose_graph_solver::findNearestFrame(constraints, timestamp_prev);
      auto poseConstrainIdxCurr = pose_graph_solver::findNearestFrame(constraints, timestamp_curr);

      auto posePrev = constraints[poseConstrainIdxPrev].realigned_pose;
      auto poseCurr = constraints[poseConstrainIdxCurr].realigned_pose;
      auto rel = posePrev.inverse() * poseCurr;

      auto quat_rel = Eigen::Quaterniond(rel.unit_quaternion().w(),
                                         rel.unit_quaternion().x(),
                                         rel.unit_quaternion().y(),
                                         rel.unit_quaternion().z());
      auto t_rel = rel.translation();

      ceres::examples::Pose3d newPoseConstraint;
      newPoseConstraint.q = quat_rel;
      newPoseConstraint.p = t_rel;

      ceres::examples::Constraint3d constraint;
      constraint.prev = timestamp_prev;
      constraint.curr = timestamp_curr;
      constraint.t_be = newPoseConstraint;
      constraintsCeres_gps_rel.push_back(constraint);
    }
  }

  // build a problem and solve
  ceres::Problem problem;
  pose_graph_solver::BuildOptimizationProblem(constraintsCeres, constraintsCeres_poses,
                                              constraintsCeres_gps_rel,
                                              &poses_frames,
                                              &poses_gps, &problem, frame_poses.rbegin()->first);
  bool res = pose_graph_solver::SolveOptimizationProblem(&problem);

  // if solve was successful, update the currrent SLAM pose
  if (res)
  {
    for (auto& pair : frame_poses)
    {
      if (pair.first == frame_poses.rbegin()->first)
      {
        ceres::examples::Pose3d newPose = poses_frames[pair.first];
        Sophus::SE3d newSophusPose(newPose.q, newPose.p);

        bool is_linearised = pair.second.isLinearized();
        pair.second = PoseStateWithLin(pair.first, newSophusPose);
        if (is_linearised)
        {
          pair.second.setLinTrue();
        }
      }
    }
  }
}

void KeypointVoEstimator::addVisionToQueue(const OpticalFlowResult::Ptr& data) {
  vision_data_queue->push(data);
}

void KeypointVoEstimator::puplishData(
    const OpticalFlowResult::Ptr& opt_flow_meas)
{
  GRANITE_ASSERT(frame_states.empty());
  std::vector<FrameId> frames_t_ns;
  Eigen::aligned_vector<Sophus::SE3d> frames;
  for (const auto& kv : frame_poses)
  {
    if (allFrames.count(kv.first) == 0)
    {
      allFrames[kv.first] = kv.second.getPose();
      all_poses[kv.first] = kv.second.getPose();
    }
    frames_t_ns.push_back(kv.first);
    frames.emplace_back(kv.second.getPose());
  }

  if (out_state_queue)
  {
    VioStateData::Ptr data(new VioStateData());

    const PoseStateWithLin& p = frame_poses.crbegin()->second;

    data->state =
        PoseVelBiasState(this_state_t_ns, p.getPose(), Eigen::Vector3d::Zero(),
                         Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

    data->map_idx = map_idx;

    if (calib.T_i_c.size() > 1) {
      auto sv = stereoScaleUncertainty();
      if (sv) data->scale_variance = sv.value();
    } else {
      auto sd = monoScaleDriftVariance();
      if (sd) data->drift_variance = sd.value();
    }

    data->frames_t_ns = frames_t_ns;
    data->frames = frames;

    data->order = optim_order;
    data->H = optim_H;
    data->b = optim_b;

    out_state_queue->push(data);
  }

  // frames added here
  if (out_vis_queue)
  {
    VioVisualizationData::Ptr data(new VioVisualizationData);

    data->t_ns = this_state_t_ns;

    data->map_idx = map_idx;

    data->frames_t_ns = frames_t_ns;
    data->frames = frames;

    get_current_points(data->points, data->point_ids);

    data->projections.resize(opt_flow_meas->observations.size());
    computeProjections(data->projections);

    data->opt_flow_res = opt_flow_meas;

    data->negative_entropy_last_frame = negative_entropy_last_frame;

    data->average_negative_entropy_last_frame =
        average_negative_entropy_last_frame;

    data->take_kf = take_kf;

    out_vis_queue->push(data);
  }
};

bool KeypointVoEstimator::processFrame(
    const OpticalFlowResult::Ptr& opt_flow_meas) {
  GRANITE_ASSERT_MSG(
      tracking_state != TrackingState::UNINITIALIZED,
      "You tried to process a frame with an uninitialized system.");

  num_processed_frames++;
  frames_after_kf++;

  // measure frame
  std::map<int64_t, int> num_points_connected;

  num_points_connected = measure(opt_flow_meas);

  // detect failure
  const int num_points_connected_total = std::accumulate(
      std::begin(num_points_connected), std::end(num_points_connected), 0,
      [](const int previous, const std::pair<const int64_t, int>& p)
      {
        return previous + p.second;
      });

  double dist = 0.0;
  auto idx1 = frame_poses.rbegin()->first;
  auto pc_idx1 = pose_graph_solver::findNearestFrame(poseConstraints, idx1);
  auto pc_pose = poseConstraints[pc_idx1].realigned_pose;
  if (poseConstraints[0].realigned)
  {
    dist = (frame_poses.rbegin()->second.getPose().inverse() * pc_pose).translation().norm();
  }

  double threshold = 2.0;

  /// i adding this condition because it seems that at the end of the sequence i have a some disconvergence between
  /// GPS pose and visual poses, so its is better to not rely on this data anymore.
  bool lost = (dist > threshold) && (allFramesCtr < 1400);

  // TODO address magic number
  const size_t min_keyframes = calib.stereo_pairs.size() == 0 ? 2 : 1;
  if ((tracking_state == TrackingState::TRACKING &&
      kf_ids.size() >= min_keyframes && num_points_connected_total < 7) || lost)
  {
    // tracking -------> LOST
    tracking_state = TrackingState::LOST;
    allFrames.clear();

    const FrameId this_state_t_ns_save = this_state_t_ns;

    reset();
    map_idx++;

    if (lost)
    {
      init_first_pose(this_state_t_ns_save, pc_pose);
    }
    else
    {
      init_first_pose(this_state_t_ns_save, Sophus::SE3d());
    }

    num_points_connected = measure(opt_flow_meas);

    puplishData(opt_flow_meas);
  }
  else
  {
    puplishData(opt_flow_meas);
    marginalize(num_points_connected);
  }

  // output



  // prepare next step
  prev_state_t_ns = this_state_t_ns;
  last_processed_t_ns = this_state_t_ns;
  if (take_kf)
  {
    average_negative_entropy_last_frame = 0;
  }

  last_unconnected_obs = std::move(this_unconnected_obs);
  last_untriangulated_obs = std::move(this_untriangulated_obs);

  take_kf = false;

  return true;
}



std::map<int64_t, int> KeypointVoEstimator::measure(
    const OpticalFlowResult::Ptr& opt_flow_meas)
{
  GRANITE_ASSERT_MSG(tracking_state == TrackingState::TRACKING,
                    "You tried to track a frame but the system state is not "
                    "TRACKING. Try initializing it first.");

  // save results
  prev_opt_flow_res[opt_flow_meas->t_ns] = opt_flow_meas;
  //prev_opt_flow_res[opt_flow_meas->t_ns].

  // Make new residual for existing keypoints
  int connected_finite = 0;
  std::map<int64_t, int> num_points_connected;
  std::map<int64_t, int> num_finite_points_connected;
  for (CamId i = 0; i < opt_flow_meas->observations.size(); i++) {
    TimeCamId tcid_target(opt_flow_meas->t_ns, i);

    bool is_main_cam = std::count(calib.main_cam_idx.begin(),
                                  calib.main_cam_idx.end(), i) != 0;

    for (const auto& kv_obs : opt_flow_meas->observations[i])
    {
      KeypointId kpt_id = kv_obs.first;

      if (lmdb.landmarkExists(kpt_id))
      {
        const KeypointPosition& kpt_pos = lmdb.getLandmark(kpt_id);
        const TimeCamId& tcid_host = kpt_pos.kf_id;

        KeypointObservation kobs;
        kobs.kpt_id = kpt_id;
        kobs.pos = kv_obs.second.translation().cast<double>();
        lmdb.addObservation(tcid_target, kobs);

        num_points_connected[tcid_host.frame_id]++;
        if (kpt_pos.id == 0) {
          if (is_main_cam) {
            this_untriangulated_obs[tcid_target.cam_id].emplace(kpt_id);
          }
        } else {
          num_finite_points_connected[tcid_host.frame_id]++;
          if (is_main_cam) connected_finite++;
        }
      } else {
        if (is_main_cam) {
          this_unconnected_obs[tcid_target.cam_id].emplace(kpt_id);
        }
      }
    }
  }

  optimize(num_points_connected, connected_finite);

  return num_points_connected;
}

void KeypointVoEstimator::create_keyframe(
    std::map<int64_t, int>& num_points_connected, int connected_finite) {
  int connected_infinite = std::accumulate(
      std::begin(last_untriangulated_obs), std::end(last_untriangulated_obs), 0,
      [](const int previous, const auto& p) {
        return previous + p.second.size();
      });

  // first frame should become a keyframe immediately
  const int64_t newkf_t_ns =
      prev_state_t_ns < 0 ? this_state_t_ns : prev_state_t_ns;
  std::unordered_map<CamId, std::unordered_set<int>>& newkf_unconnected_obs =
      prev_state_t_ns < 0 ? this_unconnected_obs : last_unconnected_obs;
  std::unordered_map<CamId, std::unordered_set<int>>& newkf_untriangulated_obs =
      prev_state_t_ns < 0 ? this_untriangulated_obs : last_untriangulated_obs;

  if (kf_ids.count(newkf_t_ns) > 0) {
    return;
  }

  // make keyframe
  std::cout << "Taking keyframe: " << newkf_t_ns << std::endl;
  kf_ids.emplace(newkf_t_ns);
  frames_after_kf = 0;

  // TODO address magic number
  if (connected_finite < 5 && prev_state_t_ns >= 0) {
    if (map_initialized) {
      map_idx++;
    }

    map_initialized = false;
    // looks like we have almost no triangulated points in the map
    // try a relative pose initialization
    rel_pose_initialisation(num_points_connected);
  } else {
    map_initialized = true;
  }

  int num_points_added = 0;

  const double min_triang_distance2 =
      config.vio_min_triangulation_dist * config.vio_min_triangulation_dist;

  auto find_all_observations = [&](const int lm_id,
                                   const TimeCamId& tcid_newkf) {
    std::map<TimeCamId, KeypointObservation> kp_obs;
    for (const auto& kv_opt_flow_res : prev_opt_flow_res) {
      if (kv_opt_flow_res.first != tcid_newkf.frame_id) {
        auto it = kv_opt_flow_res.second->observations.at(tcid_newkf.cam_id)
                      .find(lm_id);
        if (it !=
            kv_opt_flow_res.second->observations.at(tcid_newkf.cam_id).end()) {
          TimeCamId tcido(kv_opt_flow_res.first, tcid_newkf.cam_id);
          KeypointObservation kobs;
          kobs.kpt_id = lm_id;
          kobs.pos = it->second.translation().cast<double>();

          kp_obs[tcido] = kobs;
        }
      }

      for (const auto& stereo_pair : calib.stereo_pairs) {
        if (stereo_pair.first == tcid_newkf.cam_id) {
          auto it = kv_opt_flow_res.second->observations.at(stereo_pair.second)
                        .find(lm_id);
          if (it != kv_opt_flow_res.second->observations.at(stereo_pair.second)
                        .end()) {
            TimeCamId tcido(kv_opt_flow_res.first, stereo_pair.second);
            KeypointObservation kobs;
            kobs.kpt_id = lm_id;
            kobs.pos = it->second.translation().cast<double>();

            kp_obs[tcido] = kobs;
          }
        }
      }
    }
    return kp_obs;
  };

  // Triangulate new points
  for (const auto& luo : newkf_unconnected_obs) {
    TimeCamId tcid_newkf(newkf_t_ns, luo.first);

    for (const int lm_id : luo.second) {
      if (lmdb.landmarkExists(lm_id)) continue;

      // if (prev_opt_flow_res.at(newkf_t_ns)
      //         ->observations.at(0)
      //         .count(lm_id) == 0)
      //   continue;

      const Eigen::Vector2d p_newkf_img_plane =
          prev_opt_flow_res.at(newkf_t_ns)
              ->observations.at(luo.first)
              .at(lm_id)
              .translation()
              .cast<double>();

      const double weight = 1.0 / (1 << prev_opt_flow_res.at(newkf_t_ns)
                                            ->pyramid_levels.at(luo.first)
                                            .at(lm_id));

      Eigen::Vector4d p_newkf_3d;
      bool valid = calib.intrinsics.at(luo.first).unproject(p_newkf_img_plane,
                                                            p_newkf_3d);
      if (!valid) continue;

      KeypointObservation kobs_newkf;
      kobs_newkf.kpt_id = lm_id;
      kobs_newkf.pos = p_newkf_img_plane;

      std::map<TimeCamId, KeypointObservation> kp_obs =
          find_all_observations(lm_id, tcid_newkf);

      if (kp_obs.empty()) continue;

      // triangulate
      double best_baseline = -1;
      Eigen::Vector4d best_p_triangulated;

      for (const auto& kv_obs : kp_obs) {
        TimeCamId tcid_other = kv_obs.first;

        const Eigen::Vector2d po = prev_opt_flow_res.at(tcid_other.frame_id)
                                       ->observations.at(tcid_other.cam_id)
                                       .at(lm_id)
                                       .translation()
                                       .cast<double>();

        Eigen::Vector4d po_3d;
        bool valid2 = calib.intrinsics[tcid_other.cam_id].unproject(po, po_3d);
        if (!valid2) continue;

        const Sophus::SE3d T_inewkf_iother =
            getPoseStateWithLin(tcid_newkf.frame_id).getPose().inverse() *
            getPoseStateWithLin(tcid_other.frame_id).getPose();
        const Sophus::SE3d T_cnewkf_cother =
            calib.T_i_c.at(tcid_newkf.cam_id).inverse() * T_inewkf_iother *
            calib.T_i_c.at(tcid_other.cam_id);

        Eigen::Vector4d p_triangulated;
        const double baseline = T_cnewkf_cother.translation().squaredNorm();

        // TODO instead of using a threshold on the baseline one could look at
        // the angle between bearing vectors
        if (baseline < min_triang_distance2) {
          p_triangulated = p_newkf_3d;
        } else {
          p_triangulated = triangulate(p_newkf_3d.head<3>(), po_3d.head<3>(),
                                       T_cnewkf_cother);

          // indicator for outlier
          if (!(p_triangulated.allFinite() && p_triangulated[3] >= 0 &&
                p_triangulated[3] < config.vio_max_inverse_distance))
            continue;
        }

        // Only triangulate if numerically stable
        // TODO address magic number
        if (p_triangulated[3] <
            std::numeric_limits<double>::epsilon() * 100.0) {
          // insert as point at infinity
          p_triangulated[3] = 0;
        }

        if (baseline > best_baseline) {
          best_baseline = baseline;
          best_p_triangulated = p_triangulated;
        }
      }

      if (best_baseline >= 0) {
        KeypointPosition kpt_pos;
        kpt_pos.kf_id = tcid_newkf;
        kpt_pos.dir = StereographicParam<double>::project(best_p_triangulated);
        kpt_pos.id = best_p_triangulated[3];
        kpt_pos.weight = weight;
        lmdb.addLandmark(lm_id, kpt_pos);

        num_points_added++;
        if (best_p_triangulated[3] > 0.0) {
          connected_finite++;
        } else {
          connected_infinite++;
        };

        lmdb.addObservation(tcid_newkf, kobs_newkf);
        for (const auto& kv_obs : kp_obs) {
          lmdb.addObservation(kv_obs.first, kv_obs.second);

          if (kv_obs.first.frame_id == this_state_t_ns) {
            num_points_connected[tcid_newkf.frame_id]++;
          }
        }
      }
    }
  }
  num_points_kf[newkf_t_ns] = num_points_added;

  // triangulate existing points
  int num_points_triangulated = 0;
  for (const auto& luo : newkf_untriangulated_obs) {
    TimeCamId tcid_newkf(newkf_t_ns, luo.first);
    for (const auto lm_id : luo.second) {
      const Eigen::Vector2d p_newkf_img_plane =
          prev_opt_flow_res.at(newkf_t_ns)
              ->observations.at(tcid_newkf.cam_id)
              .at(lm_id)
              .translation()
              .cast<double>();

      const double weight = 1.0 / (1 << prev_opt_flow_res.at(newkf_t_ns)
                                            ->pyramid_levels.at(luo.first)
                                            .at(lm_id));

      Eigen::Vector4d p_newkf_3d;
      const bool valid1 = calib.intrinsics.at(tcid_newkf.cam_id)
                              .unproject(p_newkf_img_plane, p_newkf_3d);
      if (!valid1) continue;

      KeypointObservation kobs_cam0_newkf;
      kobs_cam0_newkf.kpt_id = lm_id;
      kobs_cam0_newkf.pos = p_newkf_img_plane;

      std::map<TimeCamId, KeypointObservation> kp_obs =
          find_all_observations(lm_id, tcid_newkf);

      if (kp_obs.empty()) continue;

      double best_baseline = -1;
      Eigen::Vector4d best_p_triangulated;
      for (const auto& kv_obs : kp_obs) {
        const TimeCamId tcid_other = kv_obs.first;

        Eigen::Vector4d po_3d;
        bool valid2 = calib.intrinsics.at(tcid_other.cam_id)
                          .unproject(kv_obs.second.pos, po_3d);
        if (!valid2) continue;

        const Sophus::SE3d T_inow_iother =
            getPoseStateWithLin(tcid_newkf.frame_id).getPose().inverse() *
            getPoseStateWithLin(tcid_other.frame_id).getPose();
        const Sophus::SE3d T_cnow_cother =
            calib.T_i_c.at(tcid_newkf.cam_id).inverse() * T_inow_iother *
            calib.T_i_c.at(tcid_other.cam_id);

        const double baseline = T_cnow_cother.translation().squaredNorm();

        if (baseline < min_triang_distance2) continue;

        Eigen::Vector4d p_triangulated =
            triangulate(p_newkf_3d.head<3>(), po_3d.head<3>(), T_cnow_cother);

        // Only triangulate if numerically stable
        if (!(p_triangulated.allFinite() &&
              p_triangulated[3] >=
                  std::numeric_limits<double>::epsilon() * 100 &&
              p_triangulated[3] < config.vio_max_inverse_distance)) {
          continue;
        }

        if (baseline > best_baseline) {
          best_baseline = baseline;
          best_p_triangulated = p_triangulated;
        }
      }

      if (best_baseline > 0) {
        // could have been removed as outlier during optimization
        if (lmdb.landmarkExists(lm_id)) {
          auto& kpt_pos = lmdb.getLandmark(lm_id);
          // std::cout << "Triangulated exisiting point" << std::endl;
          const Sophus::SE3d T_h_cnow =
              calib.T_i_c.at(kpt_pos.kf_id.cam_id).inverse() *
              getPoseStateWithLin(kpt_pos.kf_id.frame_id).getPose().inverse() *
              getPoseStateWithLin(tcid_newkf.frame_id).getPose() *
              calib.T_i_c.at(tcid_newkf.cam_id);

          Eigen::Vector4d p_h = T_h_cnow.matrix() * best_p_triangulated;

          kpt_pos.dir = StereographicParam<double>::project(p_h);
          kpt_pos.id = p_h[3];
          num_points_triangulated++;
        } else {
          KeypointPosition kpt_pos;
          kpt_pos.kf_id = tcid_newkf;
          kpt_pos.dir =
              StereographicParam<double>::project(best_p_triangulated);
          kpt_pos.id = best_p_triangulated[3];
          kpt_pos.weight = weight;
          lmdb.addLandmark(lm_id, kpt_pos);
          num_points_kf[newkf_t_ns]++;
        }

        auto ex_obs = lmdb.getObservationsOfLandmark(lm_id);

        if (ex_obs.count(tcid_newkf) == 0) {
          lmdb.addObservation(tcid_newkf, kobs_cam0_newkf);
        }
        for (const auto& kv_obs : kp_obs) {
          if (ex_obs.count(kv_obs.first) == 0) {
            lmdb.addObservation(kv_obs.first, kv_obs.second);
            if (kv_obs.first.frame_id == this_state_t_ns) {
              num_points_connected[tcid_newkf.frame_id]++;
            }
          }
        }
      }
    }
  }
  // std::cout << "Triangulated " << num_points_triangulated << " points."
  //           << std::endl;
}

void KeypointVoEstimator::rel_pose_initialisation(
    std::map<int64_t, int>& num_points_connected) {
  // try to estimate the pose relative to
  const FrameId new_keyframe_t_ns = prev_state_t_ns;

  // I want to use the average of the estimated visual position and the GPS position because
  // the first alignment is very rough and can be quite far from the estimated visual
  // pose and can blow up the solution, so make it smoother by using the average
  // of the visual and GPS.
  Eigen::Matrix<double, 3, 1> mean_position;

  auto pose_constraint_idx = pose_graph_solver::findNearestFrame(poseConstraints, new_keyframe_t_ns);
  if (poseConstraints[0].realigned)
  {
    mean_position = (poseConstraints[pose_constraint_idx].realigned_pose.translation() +
           frame_poses.at(new_keyframe_t_ns).getPose().translation()) * 0.5;
  }

  auto hintPose = poseConstraints[pose_constraint_idx].realigned ? poseConstraints[pose_constraint_idx].realigned_pose :
                                              frame_poses.at(new_keyframe_t_ns).getPose();
  if (poseConstraints[0].realigned)
  {
    hintPose.translation() = mean_position;
  }

  Eigen::aligned_map<FrameId, Sophus::SE3d> current_T_newkfc0_prevc0s;
  const Sophus::SE3d current_T_c0_w =
      (hintPose * calib.T_i_c.at(0))
          .inverse();
  for (const auto& fp_kv : frame_poses) {
    current_T_newkfc0_prevc0s[fp_kv.first] =
        current_T_c0_w * fp_kv.second.getPose() * calib.T_i_c.at(0);
  }

  auto PICE = ParallelInitializationCandidateEvaluation(
      prev_opt_flow_res[new_keyframe_t_ns], prev_opt_flow_res,
      current_T_newkfc0_prevc0s, calib.intrinsics[0], calib.intrinsics[0],
      config.vio_mono_init_max_ransac_iter, config.vio_mono_init_min_parallax);
  tbb::parallel_reduce(tbb::blocked_range<size_t>(0, prev_opt_flow_res.size()),
                       PICE);
  const InitializationResult best_init_res = PICE.best_init_res;

  const double min_triang_distance2 =
      config.vio_min_triangulation_dist * config.vio_min_triangulation_dist;

  if (best_init_res.score > 0 &&
      best_init_res.T_a_b.translation().squaredNorm() > min_triang_distance2) {
    std::cout << "best_T_prev_now:\n"
              << best_init_res.T_a_b.inverse().translation() << std::endl;

    std::cout << "Will initialize between " << best_init_res.tcid_b.frame_id
              << " and " << best_init_res.tcid_a.frame_id << std::endl;

    const Sophus::SE3d T_iprev_inewkf =
        (calib.T_i_c.at(best_init_res.tcid_a.cam_id) * best_init_res.T_a_b *
         calib.T_i_c.at(best_init_res.tcid_b.cam_id).inverse())
            .inverse();

    rel_translation_constraints[FramePair(best_init_res.tcid_b.frame_id,
                                          best_init_res.tcid_a.frame_id)] =
        T_iprev_inewkf.translation().norm();

    PoseStateWithLin newkf_state(
        new_keyframe_t_ns,
        getPoseStateWithLin(best_init_res.tcid_b.frame_id).getPose() *
            T_iprev_inewkf);
    frame_poses[new_keyframe_t_ns] = newkf_state;

    // remove unneeded keyframes
    std::set<FrameId> kf_to_remove;
    for (auto& pose : frame_poses) {
      pose.second.setLinFalse();

      if (kf_ids.count(pose.first) != 0 &&
          pose.first != best_init_res.tcid_a.frame_id &&
          pose.first != best_init_res.tcid_b.frame_id) {
        size_t connected_obs = 0;
        for (KeypointId lm_id : best_init_res.triangulated_idxs) {
          for (auto obs_pair :
               prev_opt_flow_res.at(pose.first)->observations.at(0)) {
            if (lm_id == obs_pair.first) {
              connected_obs++;
            }
          }
        }
        // TODO address magic number
        if (connected_obs < 10) {
          kf_to_remove.emplace(pose.first);
        }
      }
    }

    for (const FrameId id : kf_to_remove) {
      kf_ids.erase(id);
      frame_poses.erase(id);
      prev_opt_flow_res.erase(id);
      eraseRelTranslationConstraints(id);
      num_points_kf.erase(id);
      num_points_connected.erase(id);
    }
    lmdb.removeKeyframes(kf_to_remove, kf_to_remove, kf_to_remove);

    // reset marginalization
    frame_poses.at(*kf_ids.cbegin()).setLinTrue();
    marg_H.setZero(se3_SIZE, se3_SIZE);
    marg_b.setZero(se3_SIZE);

    // prior on pose
    marg_H.diagonal().setConstant(config.vio_init_pose_weight);

    marg_order.abs_order_map.clear();
    marg_order.abs_order_map[*kf_ids.cbegin()] = std::make_pair(0, se3_SIZE);
    marg_order.total_size = se3_SIZE;
    marg_order.items = 1;

    map_initialized = true;
  }
}

void KeypointVoEstimator::checkMargNullspace() const {
  checkNullspace(marg_H, marg_b, marg_order, frame_states, frame_poses);
}

void KeypointVoEstimator::marginalize(
    std::map<FrameId, int>& num_points_connected)
{
  //double weight = 1;
  GRANITE_ASSERT(frame_states.empty());

  if (true)
  {
    // Marginalize

    AbsOrderMap aom;

    // remove all frame_poses that are not kfs and not the current frame
    std::set<FrameId> non_kf_poses;
    for (const auto& kv : frame_poses)
    {
      if (kf_ids.count(kv.first) == 0 && kv.first != this_state_t_ns)
      {
        non_kf_poses.emplace(kv.first);
      }
      else
      {
        if (kv.first != this_state_t_ns)
        {
          aom.abs_order_map[kv.first] =
              std::make_pair(aom.total_size, se3_SIZE);

          // Check that we have the same order as marginalization
          if (marg_order.abs_order_map.count(kv.first) > 0)
            GRANITE_ASSERT(marg_order.abs_order_map.at(kv.first) ==
                          aom.abs_order_map.at(kv.first));

          aom.total_size += se3_SIZE;
          aom.items++;
        }
      }
    }

    for (int64_t id : non_kf_poses)
    {
      frame_poses.erase(id);
      lmdb.removeFrame(id);
      prev_opt_flow_res.erase(id);
    }

    auto kf_ids_all = kf_ids;
    std::set<FrameId> kfs_to_marg;
    while (kf_ids.size() > max_kfs)
    {
      int64_t id_to_marg = -1;
      {
        std::vector<int64_t> ids;
        for (int64_t id : kf_ids)
        {
          ids.push_back(id);
        }

        for (size_t i = 0; i < ids.size() - 2; i++)
        {
          if (num_points_connected.count(ids[i]) == 0 ||
              num_points_kf.at(ids[i]) == 0 ||
              (num_points_connected.at(ids[i]) / num_points_kf.at(ids[i]) <
               0.05))
          {
            id_to_marg = ids[i];
            break;
          }
        }
      }

      if (id_to_marg < 0) {
        std::vector<int64_t> ids;
        for (int64_t id : kf_ids)
        {
          ids.push_back(id);
        }

        int64_t last_kf = *kf_ids.crbegin();
        double min_score = std::numeric_limits<double>::max();
        int64_t min_score_id = -1;

        for (size_t i = 0; i < ids.size() - 2; i++)
        {
          double denom = 0;
          for (size_t j = 0; j < ids.size() - 2; j++)
          {
            denom += 1 / ((frame_poses.at(ids[i]).getPose().translation() -
                           frame_poses.at(ids[j]).getPose().translation())
                              .norm() +
                          1e-5);
          }

          double score =
              std::sqrt((frame_poses.at(ids[i]).getPose().translation() -
                         frame_poses.at(last_kf).getPose().translation())
                            .norm()) *
              denom;

          if (score < min_score)
          {
            min_score_id = ids[i];
            min_score = score;
          }
        }

        id_to_marg = min_score_id;
      }

      kfs_to_marg.emplace(id_to_marg);
      non_kf_poses.emplace(id_to_marg);

      kf_ids.erase(id_to_marg);
    }

    if (!kfs_to_marg.empty())
    {
      // Marginalize only if last state is a keyframe
      // GRANITE_ASSERT(kf_ids_all.count(this_state_t_ns) > 0);
      GRANITE_ASSERT(kf_ids_all.count(prev_state_t_ns) > 0);

      size_t asize = aom.total_size;
      double marg_prior_error;

      DenseAccumulator accum;
      accum.reset(asize);

      {
        // Linearize points

        Eigen::aligned_map<
            TimeCamId,
            Eigen::aligned_map<TimeCamId,
                               Eigen::aligned_vector<KeypointObservation>>>
            obs_to_lin;

        for (auto it = lmdb.getObservations().cbegin();
             it != lmdb.getObservations().cend();)
        {
          if (kfs_to_marg.count(it->first.frame_id) > 0)
          {
            for (auto it2 = it->second.cbegin(); it2 != it->second.cend();
                 ++it2)
            {
              obs_to_lin[it->first].emplace(*it2);
            }
          }
          ++it;
        }

        double rld_error;
        Eigen::aligned_vector<RelLinData<se3_SIZE>> rld_vec;

        linearizeHelper(rld_vec, obs_to_lin, rld_error);

        for (auto& rld : rld_vec)
        {
          rld.invert_keypoint_hessians();

          Eigen::MatrixXd rel_H;
          Eigen::VectorXd rel_b;
          linearizeRel(rld, rel_H, rel_b);

          linearizeAbs(rel_H, rel_b, rld, aom, accum);
        }
      }

      // rel_translation_pose_constraints
      double rel_pose_constraint_error = 0;
      //double rel_pose_constraint_pc_error = 0;
/*      linearizeRelTranslationConstraints(
          aom, accum.getH(), accum.getB(), rel_pose_constraint_pc_error,
          config.vio_init_pose_weight, rel_translation_pose_constraints,
          frame_poses);*/

/*      linearizeRelTranslationConstraintsRelSE3(accum.getH(), accum.getB(),
                                               rel_pose_constraint_pc_error, weight,
                                               rel_translation_pose_constraints, frame_poses);*/
      linearizeRelTranslationConstraints(
          aom, accum.getH(), accum.getB(), rel_pose_constraint_error,
          config.vio_init_pose_weight, rel_translation_constraints,
          frame_poses);


      linearizeMargPrior(marg_order, marg_H, marg_b, aom, accum.getH(),
                         accum.getB(), marg_prior_error);

      // Save marginalization prior
      if (out_marg_queue && !kfs_to_marg.empty())
      {
        {
          MargData::Ptr m(new MargData);
          m->aom = aom;
          m->abs_H = accum.getH();
          m->abs_b = accum.getB();
          m->frame_poses = frame_poses;
          m->frame_states = frame_states;
          m->kfs_all = kf_ids_all;
          m->kfs_to_marg = kfs_to_marg;
          m->use_imu = false;

          for (int64_t t : m->kfs_all) {
            m->opt_flow_res.emplace_back(prev_opt_flow_res.at(t));
          }

          out_marg_queue->push(m);
        }
      }

      std::set<int> idx_to_keep, idx_to_marg;
      for (const auto& kv : aom.abs_order_map) {
        if (kv.second.second == se3_SIZE) {
          int start_idx = kv.second.first;
          if (kfs_to_marg.count(kv.first) == 0) {
            for (size_t i = 0; i < se3_SIZE; i++)
              idx_to_keep.emplace(start_idx + i);
          } else {
            for (size_t i = 0; i < se3_SIZE; i++)
              idx_to_marg.emplace(start_idx + i);
          }
        } else {
          GRANITE_ASSERT(false);
        }
      }

      Eigen::MatrixXd marg_H_new;
      Eigen::VectorXd marg_b_new;
      marginalizeHelper(accum.getH(), accum.getB(), idx_to_keep, idx_to_marg,
                        marg_H_new, marg_b_new);

      for (auto& kv : frame_poses)
      {
        if (kv.first != this_state_t_ns && !kv.second.isLinearized())
          kv.second.setLinTrue();
      }

      for (const FrameId id : kfs_to_marg)
      {
        frame_poses.erase(id);
        prev_opt_flow_res.erase(id);
        eraseRelTranslationConstraints(id);
        num_points_kf.erase(id);
        num_points_connected.erase(id);
      }

      lmdb.removeKeyframes(kfs_to_marg, kfs_to_marg, kfs_to_marg);

      // constrain global position
      marg_H_new.topLeftCorner<6, 6>() +=
          Eigen::Matrix<double, 6, 6>::Identity() * config.vio_init_pose_weight;

      AbsOrderMap marg_order_new;

      for (const auto& kv : frame_poses) {
        if (kf_ids.count(kv.first) > 0) {
          marg_order_new.abs_order_map[kv.first] =
              std::make_pair(marg_order_new.total_size, se3_SIZE);

          marg_order_new.total_size += se3_SIZE;
          marg_order_new.items++;
        }
      }

      marg_H = marg_H_new;
      marg_b = marg_b_new;
      marg_order = marg_order_new;

      GRANITE_ASSERT(size_t(marg_H.cols()) == marg_order.total_size);

      // std::cout << "marg_H\n" << marg_H << std::endl;

      Eigen::VectorXd delta;
      computeDelta(marg_order, delta);
      marg_b -= marg_H * delta;
    }
  }

  rel_translation_pose_constraints.clear();
}

void KeypointVoEstimator::optimize(std::map<int64_t, int>& num_points_connected,
                                   int connected_finite, const bool filter)
{
  granite::alignmentSe3 alignmentSe3;
  pose_graph_solver::findAlignment(all_poses, poseConstraints, alignmentSe3);
  std::cout << "alignmentSe3.singValuePercent: " << alignmentSe3.singValuePercent << std::endl;
  /*
   The idea here is that we initially use a very coarse GPS constraint, scaled by translation of visual poses.
   So we want to try to improve this rough alignment with a real one using Horn.
   This should succeed after the first rotation in the visual trajectory (since until
   we have at least one rotation, we will have nullspace in the Horn alignment
   of two straight lines). So once we've found that, rescale our constraints.
   */
  if (alignmentSe3.singValuePercent > 0.01)
  {
    if (!T_global_world.has_value())
    {
          T_global_world = alignmentSe3.aligment;
          pose_graph_solver::realignUsingEstimatedTgw(poseConstraints, T_global_world.value());
    }
  }

  allFramesCtr++;
  //double weight = 1;
  if (true)
  {
    // Optimize
    std::vector<int64_t> indexes;
    optim_order.clear();
    timestampAlignedConstraints.clear();
    for (auto& kv : frame_poses)
    {
      optim_order.abs_order_map[kv.first] =
          std::make_pair(optim_order.total_size, se3_SIZE);

      // Check that we have the same order as marginalization
      if (marg_order.abs_order_map.count(kv.first) > 0)
        GRANITE_ASSERT(marg_order.abs_order_map.at(kv.first) ==
                      optim_order.abs_order_map.at(kv.first));

      optim_order.total_size += se3_SIZE;
      optim_order.items++;
    }

    GRANITE_ASSERT(frame_states.empty());

    Eigen::MatrixXd H_keyframe_decision;
    Eigen::VectorXd b_keyframe_decision;

    int max_iter = config.vio_max_iterations;
    int filter_iter = config.vio_filter_iteration;
    for (int iter = 0; iter < max_iter + 1; iter++)
    {
      // wait until system will initialise and create some keyframes
      // so we can align our GPS data. Simply wait until system will produce 100 frames.
      if (allFrames.size() > 100 && allFramesCtr < 1400)
      {
        // we want to align only once.
        if (poseConstraints[0].realigned == false)
        {
          pose_graph_solver::realign(poseConstraints, frame_poses);
        }

        // build graph and optimise our current pose only.
        optimise_graph(frame_poses, poseConstraints);

        // update the backend part, use aligned GPS constraints information here.
        for (auto& pp : rel_translation_constraints)
        {
          auto idx1 = pose_graph_solver::findNearestFrame(poseConstraints, pp.first.frame_first);
          auto idx2 = pose_graph_solver::findNearestFrame(poseConstraints, pp.first.frame_second);

          auto p_prev = poseConstraints[idx1].realigned_pose;
          auto p_curr = poseConstraints[idx2].realigned_pose;

          rel_translation_constraints[pp.first] = (p_prev.inverse() * p_curr).inverse().translation().norm();
        }
      }

      double rld_error;
      Eigen::aligned_vector<RelLinData<se3_SIZE>> rld_vec;
      linearizeHelper(rld_vec, lmdb.getObservations(), rld_error);

      BundleAdjustmentBase::LinearizeAbsReduce<DenseAccumulator<double>, se3_SIZE> lopt(optim_order);

      tbb::blocked_range<Eigen::aligned_vector<RelLinData<se3_SIZE>>::iterator> range(rld_vec.begin(), rld_vec.end());

      tbb::parallel_reduce(range, lopt);
      double rel_pose_pc_constraint_error = 0;
      double rel_pose_constraint_error = 0;


      linearizeRelTranslationConstraints(
          optim_order, lopt.accum.getH(), lopt.accum.getB(),
          rel_pose_constraint_error, config.vio_init_pose_weight,
          rel_translation_constraints, frame_poses);

/*      linearizeRelTranslationConstraints(
          optim_order, lopt.accum.getH(), lopt.accum.getB(),
          rel_pose_pc_constraint_error, config.vio_init_pose_weight,
          rel_translation_pose_constraints, frame_poses);*/

/*      linearizeRelTranslationConstraintsRelSE3(lopt.accum.getH(), lopt.accum.getB(),
                                               rel_pose_pc_constraint_error, weight,
                                               rel_translation_pose_constraints, frame_poses);*/

      double marg_prior_error = 0;
      linearizeMargPrior(marg_order, marg_H, marg_b, optim_order,
                         lopt.accum.getH(), lopt.accum.getB(),
                         marg_prior_error);

      marg_prior_error = 0.0;

      double error_total =
          rld_error + rel_pose_constraint_error  + marg_prior_error + rel_pose_pc_constraint_error;

      lopt.accum.setup_solver();
      Eigen::VectorXd Hdiag = lopt.accum.Hdiagonal();

      bool converged = false;

      if (iter == max_iter)
      {
        optim_H = lopt.accum.getH();
        optim_b = lopt.accum.getB();
      }
      else if (iter == config.vio_take_keyframe_iteration)
      {
        H_keyframe_decision = lopt.accum.getH();
        b_keyframe_decision = lopt.accum.getB();
      }
      else
      {
        if (true)
        {  // Use Levenberg–Marquardt
          bool step = false;
          int max_lm_iter = 10;

          while (!step && max_lm_iter > 0 && !converged)
          {
            Eigen::VectorXd Hdiag_lambda = Hdiag * lambda;
            for (int i = 0; i < Hdiag_lambda.size(); i++)
            {
              Hdiag_lambda[i] = std::max(Hdiag_lambda[i], min_lambda);
            }

            const Eigen::VectorXd inc = lopt.accum.solve(&Hdiag_lambda);

            double max_inc = -1.0;
            double after_error_total = -1.0;
            double f_diff = -1.0;

            if (inc.allFinite())
            {
              max_inc = inc.array().abs().maxCoeff();
              if (max_inc < 1e-4) converged = true;

              backup();

              // apply increment to poses
              for (auto& kv : frame_poses)
              {
                int idx = optim_order.abs_order_map.at(kv.first).first;
                kv.second.applyInc(-inc.segment<se3_SIZE>(idx));
              }

              GRANITE_ASSERT(frame_states.empty());

              // Update points
              tbb::blocked_range<size_t> keys_range(0, rld_vec.size());
              auto update_points_func =
                  [&](const tbb::blocked_range<size_t>& r) {
                    for (size_t i = r.begin(); i != r.end(); ++i) {
                      const auto& rld = rld_vec[i];
                      updatePoints(optim_order, rld, inc);
                    }
                  };
              tbb::parallel_for(keys_range, update_points_func);

              double after_update_marg_prior_error = 0;
              double after_update_rel_pose_constraint_error = 0;
              double after_update_rel_pc_pose_constraint_error = 0;
              double after_update_vision_error = 0;

              computeError(after_update_vision_error);

              computeRelTranslationConstraintsError(
                  optim_order, after_update_rel_pose_constraint_error,
                  config.vio_init_pose_weight, rel_translation_constraints,
                  frame_poses);

/*              computeRelTranslationConstraintsError(
                  optim_order, after_update_rel_pc_pose_constraint_error,
                  config.vio_init_pose_weight, rel_translation_pose_constraints,
                  frame_poses);*/

/*              computeRelTranslationConstraintsErrorSE3(optim_order, after_update_rel_pc_pose_constraint_error,
                                                       weight,
                                                       rel_translation_pose_constraints,
                                                       frame_poses);*/

              computeMargPriorError(marg_order, marg_H, marg_b,
                                    after_update_marg_prior_error);

              after_update_marg_prior_error = 0;

              after_error_total = after_update_vision_error +
                                  after_update_rel_pose_constraint_error +
                                  after_update_marg_prior_error +
                                  after_update_rel_pc_pose_constraint_error;

              f_diff = (error_total - after_error_total);
            }

            if (f_diff < 0)
            {
              lambda = std::min(max_lambda, lambda_vee * lambda);
              lambda_vee *= 2;

              restore();
            }
            else
            {
              lambda = std::max(min_lambda, lambda / 3);
              lambda_vee = 2;

              step = true;
            }
            max_lm_iter--;
          }
        }
        else
        {  // Use Gauss-Newton
          Eigen::VectorXd Hdiag_lambda = Hdiag * min_lambda;
          for (int i = 0; i < Hdiag_lambda.size(); i++)
          {
            Hdiag_lambda[i] = std::max(Hdiag_lambda[i], min_lambda);
          }

          const Eigen::VectorXd inc = lopt.accum.solve(&Hdiag_lambda);

          if (inc.allFinite())
          {
            double max_inc = inc.array().abs().maxCoeff();
            if (max_inc < 1e-4) converged = true;

            // apply increment to poses
            for (auto& kv : frame_poses) {
              int idx = optim_order.abs_order_map.at(kv.first).first;
              kv.second.applyInc(-inc.segment<se3_SIZE>(idx));
            }

            // apply increment to states
            for (auto& kv : frame_states)
            {
              int idx = optim_order.abs_order_map.at(kv.first).first;
              kv.second.applyInc(-inc.segment<se3_VEL_BIAS_SIZE>(idx));
            }

            // Update points
            tbb::blocked_range<size_t> keys_range(0, rld_vec.size());
            auto update_points_func = [&](const tbb::blocked_range<size_t>& r)
            {
              for (size_t i = r.begin(); i != r.end(); ++i) {
                const auto& rld = rld_vec[i];
                updatePoints(optim_order, rld, inc);
              }
            };
            tbb::parallel_for(keys_range, update_points_func);
          }
        }
      }


      // rel_translation_pose_constraints.clear();

      if (converged && iter > config.vio_take_keyframe_iteration &&
          iter > filter_iter)
      {
        iter = max_iter - 1;
      }
      else if (iter == config.vio_take_keyframe_iteration)
      {
        // calculate the negative entropy of the current frame as described in
        // "Redesigning SLAM for Arbitrary Multi-Camera Systems" Kuo et. al.
        // 2020 to decide if the previous frame should become a keyframe
        std::set<int> idx_old, idx_last;
        for (const auto& kv : optim_order.abs_order_map)
        {
          if (kv.second.second == se3_SIZE)
          {
            int start_idx = kv.second.first;
            if (kv.first == this_state_t_ns)
            {
              if (connected_finite > 5)
              {
                for (size_t i = 0; i < 3; i++) idx_last.emplace(start_idx + i);
              }
              else
              {
                // without triangulated points, translation is unconstrained
                for (size_t i = 0; i < 3; i++) idx_old.emplace(start_idx + i);
              }
              for (size_t i = 3; i < se3_SIZE; i++)
                idx_last.emplace(start_idx + i);
            }
            else
            {
              for (size_t i = 0; i < se3_SIZE; i++)
                idx_old.emplace(start_idx + i);
            }
          }
          else
          {
            GRANITE_ASSERT(false);
          }
        }

        if (!idx_old.empty() && !idx_last.empty())
        {
          Eigen::MatrixXd H_last_state;
          Eigen::VectorXd b_last_state;
          // marginalizeHelper has side effects on H and b
          marginalizeHelper(H_keyframe_decision, b_keyframe_decision, idx_last,
                            idx_old, H_last_state, b_last_state);

          const double det = H_last_state.determinant();

          if (det > std::numeric_limits<double>::epsilon())
          {
            negative_entropy_last_frame = std::log(det);

            increment_average_negative_entropy(negative_entropy_last_frame);

            int connected_infinite =
                std::accumulate(std::begin(this_untriangulated_obs),
                                std::end(this_untriangulated_obs), 0,
                                [](const int previous, const auto& p) {
                                  return previous + p.second.size();
                                });

            // TODO address magic number
            if (negative_entropy_last_frame <=
                    config.vio_fraction_entropy_take_kf *
                        average_negative_entropy_last_frame ||
                connected_finite + connected_infinite < 20)
            {
              take_kf = true;
            }
          } else
          {
            take_kf = true;
          }
        }
        else
        {
          take_kf = true;
        }

        if (take_kf)
        {
          max_iter += 2;
          filter_iter += 2;
          create_keyframe(num_points_connected, connected_finite);
          converged = false;

          // rel_pose_initialization may have removed some kf
          if (optim_order.items != frame_poses.size())
          {
            optim_order.clear();
            for (const auto& kv : frame_poses)
             {

              optim_order.abs_order_map[kv.first] =
                  std::make_pair(optim_order.total_size, se3_SIZE);

              // Check that we have the same order as marginalization
              if (marg_order.abs_order_map.count(kv.first) > 0)
                GRANITE_ASSERT(marg_order.abs_order_map.at(kv.first) ==
                              optim_order.abs_order_map.at(kv.first));

              optim_order.total_size += se3_SIZE;
              optim_order.items++;
            }
          }
        }
      }
      else if (filter && iter == filter_iter)
      {
        filterOutliers(config.vio_outlier_threshold, 3, &num_points_connected);
      }
    }
  }
}

std::optional<double> KeypointVoEstimator::stereoScaleUncertainty() const {
  Eigen::aligned_map<int64_t, PoseStateWithLin> kf_poses;
  for (const auto& p : frame_poses) {
    // if (kf_ids.count(p.first) != 0) {
    kf_poses[p.first] = p.second;
    // }
  }

  if (kf_poses.size() < 3) {
    return {};
  }

  std::set<int> idx_marg, idx_keep;
  Eigen::MatrixXd H;
  Eigen::VectorXd b;

  // linearize points

  double rld_error;
  LinDataRelScale<sim3_SIZE> ld = {};
  linearizeHelperRelSim3(ld, kf_poses, lmdb.getObservations(), rld_error);

  // schur complement

  ld.invert_landmark_hessians();

  for (const auto& kv_l_Hpls : ld.Hpl) {
    const auto lm_id = kv_l_Hpls.first;

    const auto& Hll_inv = ld.Hll.at(lm_id);

    for (const auto& kv_i : kv_l_Hpls.second) {
      const auto rel_pose_i = kv_i.first;
      const size_t rel_pose_i_start = rel_pose_i * sim3_SIZE;

      const Eigen::Matrix<double, sim3_SIZE, 3> Hpl_Hll_inv =
          kv_i.second * Hll_inv;

      ld.bp.segment<sim3_SIZE>(rel_pose_i_start) -=
          Hpl_Hll_inv * ld.bl.at(lm_id);

      for (const auto& kv_j : kv_l_Hpls.second) {
        const auto rel_pose_j = kv_j.first;
        const size_t rel_pose_j_start = rel_pose_j * sim3_SIZE;

        ld.Hpp.block<sim3_SIZE, sim3_SIZE>(rel_pose_i_start,
                                           rel_pose_j_start) -=
            Hpl_Hll_inv * kv_j.second.transpose();
      }
    }
  }

  double translation_error;
  linearizeRelTranslationConstraintsRelSim3(
      ld.Hpp, ld.bp, translation_error, config.vio_init_pose_weight,
      rel_translation_constraints, frame_poses);

  H = ld.Hpp;
  b = ld.bp;

  // transform marginalization prior

  Eigen::aligned_map<int64_t, Sophus::Sim3d> rel_poses;
  rel_poses[kf_poses.cbegin()->first] =
      Sophus::se3_2_sim3(kf_poses.cbegin()->second.getPose());
  for (auto iter = kf_poses.cbegin(); std::next(iter) != kf_poses.cend();
       iter++) {
    rel_poses[std::next(iter)->first] = Sophus::se3_2_sim3(
        iter->second.getPose().inverse() * std::next(iter)->second.getPose());
  }

  Eigen::aligned_vector<Sophus::Sim3d> chain_i;
  for (const auto& kv_i : marg_order.abs_order_map) {
    const int64_t i_t_ns = kv_i.first;
    chain_i.emplace_back(rel_poses.at(i_t_ns));

    Eigen::aligned_vector<Sophus::Matrix7d> d_i_d_xi(chain_i.size());
    concatRelPoseSim3(chain_i, Sophus::SE3d(), Sophus::SE3d(), &d_i_d_xi);

    Eigen::aligned_vector<Sophus::Sim3d> chain_j;
    for (const auto& kv_j : marg_order.abs_order_map) {
      const int64_t j_t_ns = kv_j.first;
      chain_j.emplace_back(rel_poses.at(j_t_ns));

      Eigen::aligned_vector<Sophus::Matrix7d> d_j_d_xi(chain_j.size());
      concatRelPoseSim3(chain_j, Sophus::SE3d(), Sophus::SE3d(), &d_j_d_xi);

      Sophus::Matrix7d H_marg_block_sim3;
      H_marg_block_sim3.setZero();
      H_marg_block_sim3.topLeftCorner<se3_SIZE, se3_SIZE>() =
          marg_H.block<se3_SIZE, se3_SIZE>(kv_i.second.first,
                                           kv_j.second.first);

      Sophus::Vector7d b_marg_segment_sim3;
      b_marg_segment_sim3.setZero();
      b_marg_segment_sim3.head<se3_SIZE>() =
          marg_b.segment<se3_SIZE>(kv_i.second.first);

      for (size_t ii = 0; ii < chain_i.size() - 1; ii++) {
        b.segment<sim3_SIZE>(ii * sim3_SIZE) +=
            d_i_d_xi.at(ii + 1).transpose() * b_marg_segment_sim3;

        for (size_t jj = 0; jj < chain_j.size() - 1; jj++) {
          H.block<sim3_SIZE, sim3_SIZE>(ii * sim3_SIZE, jj * sim3_SIZE) +=
              d_i_d_xi.at(ii + 1).transpose() * H_marg_block_sim3 *
              d_j_d_xi.at(jj + 1);
        }
      }
    }
  }

  // marginalize
  int start_idx = 0;
  for (auto iter = kf_poses.cbegin(); std::next(iter) != kf_poses.cend();
       iter++) {
    for (size_t i = 0; i < sim3_SIZE - 1; i++) {
      idx_marg.emplace(start_idx + i);
    }
    idx_keep.emplace(start_idx + sim3_SIZE - 1);

    start_idx += sim3_SIZE;
  }

  if (idx_marg.empty() || idx_keep.empty()) {
    return {};
  }

  Eigen::MatrixXd H_scale;
  Eigen::VectorXd b_scale;
  marginalizeHelper(H, b, idx_keep, idx_marg, H_scale, b_scale);

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H_scale);

  const double scale_variance = 1.0 / (std::sqrt(es.eigenvalues()(0)));

  // std::cout << "scale_variance: " << scale_variance << std::endl;

  return scale_variance;
}

std::optional<double> KeypointVoEstimator::monoScaleDriftVariance() const {
  Eigen::aligned_map<int64_t, PoseStateWithLin> kf_poses;
  for (const auto& p : frame_poses) {
    if (kf_ids.count(p.first) != 0) {
      kf_poses[p.first] = p.second;
    }
  }

  if (kf_poses.size() < 3) {
    return {};
  }

  std::set<int> idx_marg, idx_keep;
  Eigen::MatrixXd H;
  Eigen::VectorXd b;

  // linearize points

  double rld_error;
  LinDataRelScale<se3_SIZE> ld = {};
  linearizeHelperRelSE3(ld, kf_poses, lmdb.getObservations(), rld_error);

  // schur complement

  ld.invert_landmark_hessians();

  for (const auto& kv_l_Hpls : ld.Hpl) {
    const auto lm_id = kv_l_Hpls.first;

    const auto& Hll_inv = ld.Hll.at(lm_id);

    for (const auto& kv_i : kv_l_Hpls.second) {
      const auto rel_pose_i = kv_i.first;
      const size_t rel_pose_i_start = rel_pose_i * se3_SIZE;

      const Eigen::Matrix<double, se3_SIZE, 3> Hpl_Hll_inv =
          kv_i.second * Hll_inv;

      ld.bp.segment<se3_SIZE>(rel_pose_i_start) -=
          Hpl_Hll_inv * ld.bl.at(lm_id);

      for (const auto& kv_j : kv_l_Hpls.second) {
        const auto rel_pose_j = kv_j.first;
        const size_t rel_pose_j_start = rel_pose_j * se3_SIZE;

        ld.Hpp.block<se3_SIZE, se3_SIZE>(rel_pose_i_start, rel_pose_j_start) -=
            Hpl_Hll_inv * kv_j.second.transpose();
      }
    }
  }

  double translation_error;
  linearizeRelTranslationConstraintsRelSE3(
      ld.Hpp, ld.bp, translation_error, config.vio_init_pose_weight,
      rel_translation_constraints, frame_poses);


/*  double translation_error2;
  linearizeRelTranslationConstraintsRelSE3(
      ld.Hpp, ld.bp, translation_error2, 1,
      rel_translation_pose_constraints, frame_poses);*/

/*  double rel_pose_pc_constraint_error;
  linearizeRelTranslationConstraintsRelSE3(ld.Hpp, ld.bp,
                                           rel_pose_pc_constraint_error,
                                           100000000.0,
                                           rel_translation_pose_constraints,
                                           frame_poses);*/

  H = ld.Hpp;
  b = ld.bp;

  // transform marginalization prior

  Eigen::aligned_map<int64_t, Sophus::SE3d> rel_poses;
  rel_poses[kf_poses.cbegin()->first] = kf_poses.cbegin()->second.getPose();
  for (auto iter = kf_poses.cbegin(); std::next(iter) != kf_poses.cend();
       iter++) {
    rel_poses[std::next(iter)->first] =
        iter->second.getPose().inverse() * std::next(iter)->second.getPose();
  }

  Eigen::aligned_vector<Sophus::SE3d> chain_i;
  for (const auto& kv_i : marg_order.abs_order_map) {
    const int64_t i_t_ns = kv_i.first;
    chain_i.emplace_back(rel_poses.at(i_t_ns));

    Eigen::aligned_vector<Sophus::Matrix6d> d_i_d_xi(chain_i.size());
    concatRelPoseSE3(chain_i, Sophus::SE3d(), Sophus::SE3d(), &d_i_d_xi);

    Eigen::aligned_vector<Sophus::SE3d> chain_j;
    for (const auto& kv_j : marg_order.abs_order_map) {
      const int64_t j_t_ns = kv_j.first;
      chain_j.emplace_back(rel_poses.at(j_t_ns));

      Eigen::aligned_vector<Sophus::Matrix6d> d_j_d_xi(chain_j.size());
      concatRelPoseSE3(chain_j, Sophus::SE3d(), Sophus::SE3d(), &d_j_d_xi);

      Sophus::Matrix6d H_marg_block = marg_H.block<se3_SIZE, se3_SIZE>(
          kv_i.second.first, kv_j.second.first);

      Sophus::Vector6d b_marg_segment =
          marg_b.segment<se3_SIZE>(kv_i.second.first);

      for (size_t ii = 0; ii < chain_i.size() - 1; ii++) {
        b.segment<se3_SIZE>(ii * se3_SIZE) +=
            d_i_d_xi.at(ii + 1).transpose() * b_marg_segment;

        for (size_t jj = 0; jj < chain_j.size() - 1; jj++) {
          H.block<se3_SIZE, se3_SIZE>(ii * se3_SIZE, jj * se3_SIZE) +=
              d_i_d_xi.at(ii + 1).transpose() * H_marg_block *
              d_j_d_xi.at(jj + 1);
        }
      }
    }
  }

  // // trans factor
  //
  // for (auto iter = kf_poses.cbegin(); std::next(iter) != kf_poses.cend();
  //      iter++) {
  //   trans_factor +=
  //       (iter->second.getPose().inverse() *
  //       std::next(iter)->second.getPose())
  //           .translation()
  //           .norm() /
  //       double(kf_poses.size() - 1);
  // }

  // marginalize
  size_t start_idx = 0;
  for (auto iter = kf_poses.cbegin(); std::next(iter) != kf_poses.cend();
       iter++) {
    for (size_t i = 0; i < 3; i++) idx_keep.emplace(start_idx + i);
    for (size_t i = 3; i < se3_SIZE; i++) idx_marg.emplace(start_idx + i);

    start_idx += se3_SIZE;
  }

  if (idx_marg.empty() || idx_keep.empty()) {
    return {};
  }

  Eigen::MatrixXd H_scale;
  Eigen::VectorXd b_scale;
  marginalizeHelper(H, b, idx_keep, idx_marg, H_scale, b_scale);

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H_scale);
  const Eigen::VectorXd smallest_eigen_vector = es.eigenvectors().col(0);
  // trans_factor = 0;
  Eigen::VectorXd weighted_trans(smallest_eigen_vector.size());
  {
    size_t i = 0;
    for (auto iter = kf_poses.cbegin(); std::next(iter) != kf_poses.cend();
         iter++) {
      const Eigen::Vector3d rel_trans =
          (iter->second.getPose().inverse() * std::next(iter)->second.getPose())
              .translation();

      weighted_trans(i) =
          std::sqrt(std::abs(smallest_eigen_vector(i))) * rel_trans(0);
      weighted_trans(i + 1) =
          std::sqrt(std::abs(smallest_eigen_vector(i + 1))) * rel_trans(1);
      weighted_trans(i + 2) =
          std::sqrt(std::abs(smallest_eigen_vector(i + 2))) * rel_trans(2);

      // for (size_t idx = 0; idx <3; idx++) {
      //   trans_factor += std::sqrt(std::abs(smallest_eigen_vector(i + idx))) *
      //   std::abs(rel_trans(idx));
      // }

      i += 3;
    }
  }

  const double drift_variance =
      1.0 / (std::sqrt(es.eigenvalues()(0)) * weighted_trans.norm());

  // std::cout << "drift_variance: " << drift_variance << std::endl;

  return drift_variance;
}

void KeypointVoEstimator::computeProjections(
    std::vector<Eigen::aligned_vector<Eigen::Vector4d>>& data) const {
  for (const auto& kv : lmdb.getObservations()) {
    const TimeCamId& tcid_h = kv.first;

    // std::cout << " --------------------- KeypointVoEstimator::computeProjections" << std::endl;

    for (const auto& obs_kv : kv.second) {
      const TimeCamId& tcid_t = obs_kv.first;

      if (tcid_t.frame_id != this_state_t_ns) continue;

      if (tcid_h != tcid_t) {
        PoseStateWithLin state_h = getPoseStateWithLin(tcid_h.frame_id);
        PoseStateWithLin state_t = getPoseStateWithLin(tcid_t.frame_id);

        Sophus::SE3d T_t_h_sophus =
            computeRelPose(state_h.getPose(), calib.T_i_c[tcid_h.cam_id],
                           state_t.getPose(), calib.T_i_c[tcid_t.cam_id]);

        Eigen::Matrix4d T_t_h = T_t_h_sophus.matrix();

        FrameRelLinData<se3_SIZE> rld;

        std::visit(
            [&](const auto& cam) {
              for (size_t i = 0; i < obs_kv.second.size(); i++) {
                const KeypointObservation& kpt_obs = obs_kv.second[i];
                const KeypointPosition& kpt_pos =
                    lmdb.getLandmark(kpt_obs.kpt_id);

                Eigen::Matrix<double, 5, 1> res;
                Eigen::Vector4d proj;

                linearizePoint(kpt_obs, kpt_pos, T_t_h, cam, res,
                               config.vio_no_motion_reg_weight, nullptr,
                               nullptr, &proj);

                proj[3] = kpt_obs.kpt_id;
                data[tcid_t.cam_id].emplace_back(proj);
              }
            },
            calib.intrinsics[tcid_t.cam_id].variant);

      } else {
        // target and host are the same
        // residual does not depend on the pose
        // it just depends on the point

        std::visit(
            [&](const auto& cam) {
              for (size_t i = 0; i < obs_kv.second.size(); i++) {
                const KeypointObservation& kpt_obs = obs_kv.second[i];
                const KeypointPosition& kpt_pos =
                    lmdb.getLandmark(kpt_obs.kpt_id);

                Eigen::Matrix<double, 5, 1> res;
                Eigen::Vector4d proj;

                linearizePoint(kpt_obs, kpt_pos, Eigen::Matrix4d::Identity(),
                               cam, res, config.vio_no_motion_reg_weight,
                               nullptr, nullptr, &proj);

                proj[3] = kpt_obs.kpt_id;
                data[tcid_t.cam_id].emplace_back(proj);
              }
            },
            calib.intrinsics[tcid_t.cam_id].variant);
      }
    }
  }
}

}  // namespace granite
