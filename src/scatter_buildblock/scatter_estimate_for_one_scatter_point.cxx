//
//
/*
  Copyright (C) 2004- 2009-11-03, Hammersmith Imanet
  Copyright (C) 2011-07-01 - 2011, Kris Thielemans
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup scatter
  \brief Implementation of stir::SingleScatterSimulation::simulate_for_one_scatter_point

  \author Charalampos Tsoumpas
  \author Pablo Aguiar
  \author Kris Thielemans


*/
#include "stir/scatter/SingleScatterSimulation.h"
#include "stir/scatter/ScatterSimulation.h"
#ifndef NDEBUG
// currently necessary for assert below
#  include "stir/VoxelsOnCartesianGrid.h"
#endif

#include "stir/round.h"
#include <math.h>
using namespace std;
START_NAMESPACE_STIR

static const float total_Compton_cross_section_511keV = ScatterSimulation::total_Compton_cross_section(511.F);

double
SingleScatterSimulation::simulate_for_one_scatter_point(const std::size_t scatter_point_num,
                                                        const unsigned det_num_A,
                                                        const unsigned det_num_B,
                                                        const float tof_position_mm)
{
  if (this->max_single_scatter_cos_angle <= 0.F) // set to negative value by set_up(), so recompute
    {
      this->max_single_scatter_cos_angle = max_cos_angle(this->template_exam_info_sptr->get_low_energy_thres(),
                                                         2.f,
                                                         this->proj_data_info_sptr->get_scanner_ptr()->get_energy_resolution());
    }

  // static const float min_energy=energy_lower_limit(lower_energy_threshold,2.,energy_resolution);

  const CartesianCoordinate3D<float>& scatter_point = this->scatt_points_vector[scatter_point_num].coord;
  const CartesianCoordinate3D<float>& detector_coord_A = this->detection_points_vector[det_num_A];
  const CartesianCoordinate3D<float>& detector_coord_B = this->detection_points_vector[det_num_B];
  // note: costheta is -cos_angle such that it is 1 for zero scatter angle
  const float costheta = static_cast<float>(-cos_angle(detector_coord_A - scatter_point, detector_coord_B - scatter_point));
  // note: costheta is identical for scatter to A or scatter to B
  // Hence, the Compton_cross_section and energy are identical for both cases as well.
  if (this->max_single_scatter_cos_angle > costheta)
    return 0;
  const float new_energy = photon_energy_after_Compton_scatter_511keV(costheta);

  const float detection_efficiency_scatter = detection_efficiency(new_energy);
  if (detection_efficiency_scatter == 0)
    return 0;

  // Distances from scatter point S to each detector in mm.
  // R_SA and R_SB are the primary quantities; rA_squared and rB_squared are derived.
  // Placed before emission integrals because both TOF (tof_shift) and scatter_ratio (1/r²) need them.
  const float R_SA = static_cast<float>(norm(scatter_point - detector_coord_A));
  const float R_SB = static_cast<float>(norm(scatter_point - detector_coord_B));
  const float rA_squared = R_SA * R_SA;
  const float rB_squared = R_SB * R_SB;

  // ===== TOF MODIFICATION 3/3: CORE CHANGE (Watson 2007) =====
  // Non-TOF: emiss = (1/R_S?²) x integral_S^det  lambda(s) ds           (cached)
  // TOF:     emiss = (1/R_S?²) x integral_S^det  eps_t[...] lambda(s) ds (not cached)
  //
  // The TOF kernel eps_t for a source at distance s from S along the ray:
  //   I^A: eps_t = Gaussian( tof_shift + s , sigma_tof )
  //   I^B: eps_t = Gaussian( tof_shift - s , sigma_tof )
  // where tof_shift = k_STIR + (R_SB - R_SA)/2
  //
  // The attenuation integrals (atten_to_detA/B) and all other terms are unchanged.
  float emiss_to_detA, emiss_to_detB;

  if (this->tof_sigma_mm > 0.f)
    {
      // tof_shift: shifts the Gaussian kernel along the ray to match the measured TOF bin.
      // Derivation: k_expected^A(s) = (R_SA - R_SB)/2 - s  =>  kernel arg = k - k_expected = tof_shift + s
      //
      // STIR convention: find_cartesian_coordinates_given_scanner_coordinates swaps coord_1/coord_2
      // when timing_pos_num < 0 (line: if (tpos < 0) std::swap(coord_1, coord_2)).
      // After the swap, detector_coord_A = canonical_det2 and detector_coord_B = canonical_det1,
      // so k_scatter_sim = c/2*(t_A - t_B) = -k_STIR = |tof_position_mm|.
      // Using std::abs() handles both cases: no-swap (tof_pos > 0) and swap (tof_pos < 0).
      const float tof_shift = std::abs(tof_position_mm) + (R_SB - R_SA) / 2.f;
      emiss_to_detA = tof_weighted_integral_over_activity_image_between_scattpoint_det(
          scatter_point, detector_coord_A, tof_shift, +1.f, this->tof_sigma_mm);
      emiss_to_detB = tof_weighted_integral_over_activity_image_between_scattpoint_det(
          scatter_point, detector_coord_B, tof_shift, -1.f, this->tof_sigma_mm);
    }
  else
    {
      // Non-TOF: use the precomputed cached integrals (unchanged path)
      emiss_to_detA = cached_integral_over_activity_image_between_scattpoint_det(
          static_cast<unsigned int>(scatter_point_num), det_num_A);
      emiss_to_detB = cached_integral_over_activity_image_between_scattpoint_det(
          static_cast<unsigned int>(scatter_point_num), det_num_B);
    }

  if (emiss_to_detA == 0 && emiss_to_detB == 0)
    return 0;

  const float atten_to_detA = cached_exp_integral_over_attenuation_image_between_scattpoint_det(scatter_point_num, det_num_A);
  const float atten_to_detB = cached_exp_integral_over_attenuation_image_between_scattpoint_det(scatter_point_num, det_num_B);

  const float dif_Compton_cross_section_value = dif_Compton_cross_section(costheta, 511.F);

  const float scatter_point_mu = scatt_points_vector[scatter_point_num].mu_value;

#ifndef NDEBUG
  {
    // check if mu-value ok
    // currently terribly shift needed as in sample_scatter_points (TODO)
    const VoxelsOnCartesianGrid<float>& image
        = dynamic_cast<const VoxelsOnCartesianGrid<float>&>(*this->get_density_image_for_scatter_points_sptr());
    const CartesianCoordinate3D<float> voxel_size = image.get_voxel_size();
    const float z_to_middle = (image.get_max_index() + image.get_min_index()) * voxel_size.z() / 2.F;
    CartesianCoordinate3D<float> shifted = scatter_point;
    shifted.z() += z_to_middle;
    assert(scatter_point_mu
           == (*this->get_density_image_for_scatter_points_sptr())[this->get_density_image_for_scatter_points_sptr()
                                                                       ->get_indices_closest_to_physical_coordinates(shifted)]);
  }
#endif

  double scatter_ratio = 0;
  // ===== THIS BLOCK IS UNCHANGED BY TOF =====
  // The TOF modification is entirely inside emiss_to_detA and emiss_to_detB above.
  scatter_ratio
      = (emiss_to_detA * (1. / rB_squared) * pow(atten_to_detB, total_Compton_cross_section_relative_to_511keV(new_energy) - 1)
         + emiss_to_detB * (1. / rA_squared) * pow(atten_to_detA, total_Compton_cross_section_relative_to_511keV(new_energy) - 1))
        * atten_to_detB * atten_to_detA * scatter_point_mu * detection_efficiency_scatter;

  const CartesianCoordinate3D<float> detA_to_ring_center(0, -detector_coord_A[2], -detector_coord_A[3]);
  const CartesianCoordinate3D<float> detB_to_ring_center(0, -detector_coord_B[2], -detector_coord_B[3]);
  const float cos_incident_angle_AS = static_cast<float>(cos_angle(scatter_point - detector_coord_A, detA_to_ring_center));
  const float cos_incident_angle_BS = static_cast<float>(cos_angle(scatter_point - detector_coord_B, detB_to_ring_center));

  return scatter_ratio * cos_incident_angle_AS * cos_incident_angle_BS * dif_Compton_cross_section_value;
}

END_NAMESPACE_STIR
