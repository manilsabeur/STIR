//
//
/*
  Copyright (C) 2004- 2009, Hammersmith Imanet Ltd
  Copyright (C) 2016, UCL
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup scatter
  \brief Implementations of integrating functions in stir::ScatterEstimationByBin

  Functions calculates the integral along LOR in an image (attenuation or emission).
  (from scatter point to detector coordinate)

  \author Pablo Aguiar
  \author Charalampos Tsoumpas
  \author Kris Thielemans

  */
#include "stir/scatter/ScatterSimulation.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "stir/recon_buildblock/RayTraceVoxelsOnCartesianGrid.h"
#include "stir/numerics/erf.h"
#include <cmath>
START_NAMESPACE_STIR

float
ScatterSimulation::exp_integral_over_attenuation_image_between_scattpoint_det(const CartesianCoordinate3D<float>& scatter_point,
                                                                              const CartesianCoordinate3D<float>& detector_coord)
{
#ifndef NEWSCALE
  /* projectors work in pixel units, so convert attenuation data
     from cm^-1 to pixel_units^-1 */
  const float rescale
      = dynamic_cast<const DiscretisedDensityOnCartesianGrid<3, float>&>(*density_image_sptr).get_grid_spacing()[3] / 10;
#else
  const float rescale = 0.1F;
#endif

  return exp(-rescale * integral_between_2_points(*density_image_sptr, scatter_point, detector_coord));
}

float
ScatterSimulation::integral_over_activity_image_between_scattpoint_det(const CartesianCoordinate3D<float>& scatter_point,
                                                                       const CartesianCoordinate3D<float>& detector_coord)
{
  {
    const CartesianCoordinate3D<float> dist_vector = scatter_point - detector_coord;

    const float dist_sp1_det_squared = norm_squared(dist_vector);

    const float solid_angle_factor = std::min(static_cast<float>(_PI / 2), 1.F / dist_sp1_det_squared);

    return solid_angle_factor * integral_between_2_points(*activity_image_sptr, scatter_point, detector_coord);
  }
}

float
ScatterSimulation::integral_between_2_points(const DiscretisedDensity<3, float>& density,
                                             const CartesianCoordinate3D<float>& scatter_point,
                                             const CartesianCoordinate3D<float>& detector_coord)
{

  const VoxelsOnCartesianGrid<float>& image = dynamic_cast<const VoxelsOnCartesianGrid<float>&>(density);

  const CartesianCoordinate3D<float> voxel_size = image.get_grid_spacing();

  CartesianCoordinate3D<float> origin = image.get_origin();
  const float z_to_middle = (image.get_max_index() + image.get_min_index()) * voxel_size.z() / 2.F;
  origin.z() -= z_to_middle;
  /* TODO replace with image.get_index_coordinates_for_physical_coordinates */
  ProjMatrixElemsForOneBin lor;
  RayTraceVoxelsOnCartesianGrid(lor,
                                (scatter_point - origin) / voxel_size,  // should be in voxel units
                                (detector_coord - origin) / voxel_size, // should be in voxel units
                                voxel_size,                             // should be in mm
#ifdef NEWSCALE
                                1.F // normalise to mm
#else
                                1 / voxel_size.x() // normalise to some kind of 'pixel units'
#endif
  );
  lor.sort();
  float sum = 0; // add up values along LOR
  {
    ProjMatrixElemsForOneBin::iterator element_ptr = lor.begin();
    bool we_have_been_within_the_image = false;
    while (element_ptr != lor.end())
      {
        const BasicCoordinate<3, int> coords = element_ptr->get_coords();
        if (coords[1] >= image.get_min_index() && coords[1] <= image.get_max_index()
            && coords[2] >= image[coords[1]].get_min_index() && coords[2] <= image[coords[1]].get_max_index()
            && coords[3] >= image[coords[1]][coords[2]].get_min_index()
            && coords[3] <= image[coords[1]][coords[2]].get_max_index())
          {
            we_have_been_within_the_image = true;
            sum += image[coords] * element_ptr->get_value();
          }
        else if (we_have_been_within_the_image)
          {
            // we jump out of the loop as we are now at the other side of
            // the image
            //                                  break;
          }
        ++element_ptr;
      }
  }
  return sum;
}

// ===== NEW FUNCTION FOR TOF (Watson 2007) =====
// TOF-weighted emission integral from scatter_point toward detector_coord.
// Replaces integral_over_activity_image_between_scattpoint_det for TOF data.
//
// For a source at distance s from the scatter point S along the ray:
//   I^A: kernel = Gaussian( tof_shift + s , sigma_tof )   (sign_s = +1)
//   I^B: kernel = Gaussian( tof_shift - s , sigma_tof )   (sign_s = -1)
//
// The Gaussian is integrated analytically over each voxel path segment [u_start, u_end] cf the associated paper (I will add the ref when it is accepted)
// using erf, which is exact when lambda is constant within each voxel (as the ray tracer assumes).
//
// The 1/R² solid angle factor is applied at the end, identically to the non-TOF version ofc.
float
ScatterSimulation::tof_weighted_integral_over_activity_image_between_scattpoint_det(
    const CartesianCoordinate3D<float>& scatter_point,
    const CartesianCoordinate3D<float>& detector_coord,
    const float tof_shift,
    const float sign_s,
    const float tof_sigma_mm)
{
  // The first lines are just a copy/paste of what is happening in the original
  // function integral_between_2_points
  const VoxelsOnCartesianGrid<float>& image
      = dynamic_cast<const VoxelsOnCartesianGrid<float>&>(*activity_image_sptr);
  const CartesianCoordinate3D<float> voxel_size = image.get_grid_spacing();
  
  CartesianCoordinate3D<float> origin = image.get_origin();
  const float z_to_middle = (image.get_max_index() + image.get_min_index()) * voxel_size.z() / 2.F;
  origin.z() -= z_to_middle;

  ProjMatrixElemsForOneBin lor;
  RayTraceVoxelsOnCartesianGrid(lor,
                                (scatter_point - origin) / voxel_size,
                                (detector_coord - origin) / voxel_size,
                                voxel_size,
                                1.f / voxel_size.x());
  // If I now comment this line it works, I am not sur to understand why
  // lor.sort();

  // Precomputed factor for the erf argument: 1 / (sigma * sqrt(2))
  const float inv_sqrt2_sigma = 1.f / (tof_sigma_mm * static_cast<float>(std::sqrt(2.0)));

  float sum = 0.f;
  float u = 0.f; // cumulative path length from scatter point S, in mm
  bool inside = false;

  for (ProjMatrixElemsForOneBin::iterator element_ptr = lor.begin(); element_ptr != lor.end(); ++element_ptr)
    {
      const BasicCoordinate<3, int> coords = element_ptr->get_coords();
      if (coords[1] >= image.get_min_index() && coords[1] <= image.get_max_index()
          && coords[2] >= image[coords[1]].get_min_index() && coords[2] <= image[coords[1]].get_max_index()
          && coords[3] >= image[coords[1]][coords[2]].get_min_index()
          && coords[3] <= image[coords[1]][coords[2]].get_max_index())
        {
          inside = true;
          // Path length through this voxel in mm (element_ptr->get_value() is in pixel units)
          const float path_len_mm = element_ptr->get_value() * voxel_size.x();
          const float u_start = u;
          const float u_end = u + path_len_mm;

          // Integrate Gaussian(tof_shift + sign_s * s, sigma_tof) over [u_start, u_end].
          // Result = (1/2) * delta_erf, with direction depending on sign_s:
          //   sign_s = +1 (I^A): d/du [erf((tof_shift+u)/...)] > 0  =>  normal order
          //   sign_s = -1 (I^B): d/du [erf((tof_shift-u)/...)] < 0  =>  reversed order
          float tof_weight;
          if (sign_s > 0.f)
            {
              tof_weight = 0.5f * (erf((tof_shift + u_end) * inv_sqrt2_sigma)
                                   - erf((tof_shift + u_start) * inv_sqrt2_sigma));
            }
          else
            {
              tof_weight = 0.5f * (erf((tof_shift - u_start) * inv_sqrt2_sigma)
                                   - erf((tof_shift - u_end) * inv_sqrt2_sigma));
            }

          sum += image[coords] * tof_weight;
          u += path_len_mm;
        }
      else if (inside)
        break;
    }

  // Apply the 1/R² solid angle factor, identical to the non-TOF version
  const float rsd_squared = static_cast<float>(norm_squared(scatter_point - detector_coord));
  const float solid_angle_factor = std::min(static_cast<float>(_PI / 2), 1.f / rsd_squared);
  return solid_angle_factor * sum;
}

END_NAMESPACE_STIR
