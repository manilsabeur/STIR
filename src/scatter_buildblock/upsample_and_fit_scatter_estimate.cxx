/*
  Copyright (C) 2005 - 2011-12-31, Hammersmith Imanet Ltd
  Copyright (C) 2014, 2020 University College London
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
*/
/*!
\file
\ingroup scatter
\brief implementation of stir::ScatterEstimationByBin::upsample_and_fit_scatter_estimate

\author Charalampos Tsoumpas
\author Kris Thielemans
\author Daniel Deidda
*/

#include <iostream>
#include "stir/ProjDataInfo.h"
#include "stir/ExamInfo.h"
#include "stir/ProjDataInMemory.h"
#include "stir/inverse_SSRB.h"
#include "stir/scale_sinograms.h"
#include "stir/scatter/ScatterEstimation.h"
#include "stir/recon_buildblock/BinNormalisation.h"
#include "stir/interpolate_projdata.h"
#include "stir/utilities.h"
#include "stir/IndexRange2D.h"
#include "stir/stream.h"
#include "stir/Succeeded.h"
#include "stir/thresholding.h"
#include "stir/is_null_ptr.h"
#include "stir/warning.h"
#include "stir/error.h"
#include <boost/format.hpp>
#include "stir/ArrayFilter1DUsingConvolution.h"
#include <iostream>
#include <fstream>
#include <string>
/***********************************************************/

START_NAMESPACE_STIR

void
ScatterEstimation::upsample_and_fit_scatter_estimate(ProjData& scaled_scatter_proj_data,
                                                     const ProjData& emission_proj_data,
                                                     const ProjData& scatter_proj_data,
                                                     BinNormalisation& scatter_normalisation,
                                                     const ProjData& weights_proj_data,
                                                     const float min_scale_factor,
                                                     const float max_scale_factor,
                                                     const unsigned half_filter_width,
                                                     BSpline::BSplineType spline_type,
                                                     const bool remove_interleaving)
{
  info("upsample_and_fit_scatter_estimate: Interpolating scatter estimate to size of emission data");

  // -- UPSAMPLE FIX 1 --
  // Use scaled_scatter_proj_data's ProjDataInfo (target output format) so that the TOF structure
  // of the output is preserved: if the output is TOF, interpolated_direct_scatter is also TOF,
  // and interpolate_projdata will correctly loop over all TOF bins, previously it
  // was emission_proj_data that we used to have a buffer now we make it TOF
  shared_ptr<ProjDataInfo> interpolated_direct_scatter_proj_data_info_sptr(
      scaled_scatter_proj_data.get_proj_data_info_sptr()->clone());
  interpolated_direct_scatter_proj_data_info_sptr->reduce_segment_range(0, 0);
  ProjDataInMemory interpolated_direct_scatter(scaled_scatter_proj_data.get_exam_info_sptr(),
                                               interpolated_direct_scatter_proj_data_info_sptr);
  {
    bool actual_remove_interleaving = remove_interleaving;

    if (remove_interleaving
        && emission_proj_data.get_proj_data_info_sptr()->get_scanner_sptr()->get_scanner_geometry() != "Cylindrical")
      {
        warning("upsample_and_fit_scatter_estimate: forcing remove_interleaving to false as non-cylindrical projdata");
        actual_remove_interleaving = false;
      }
    // For TOF data, the view-doubling in make_non_interleaved_segment can trigger the
    // extend_without_wrapping fallback in extend_segment (when view coverage is neither
    // exactly 180° nor 360°), which corrupts the heap via out-of-bounds view indices.
    if (actual_remove_interleaving
        && scaled_scatter_proj_data.get_proj_data_info_sptr()->get_max_tof_pos_num() > 0)
      {
        actual_remove_interleaving = false;
      }

    interpolate_projdata(interpolated_direct_scatter, scatter_proj_data, spline_type, actual_remove_interleaving);
  }

  // now call inverse_SSRB, and normalise/scale if we need to
  if (min_scale_factor != 1 || max_scale_factor != 1 || !scatter_normalisation.is_trivial())
    {
      // -- UPSAMPLE FIX 1bis -- (1bis because it's the same as the first one)
      // Use scaled_scatter_proj_data's ProjDataInfo
      // (emission_proj_data is non-TOF but scaled_scatter_proj_data is TOF).
      ProjDataInMemory interpolated_scatter(scaled_scatter_proj_data.get_exam_info_sptr(),
                                            scaled_scatter_proj_data.get_proj_data_info_sptr()->create_shared_clone());
      inverse_SSRB(interpolated_scatter, interpolated_direct_scatter);

      scatter_normalisation.set_up(scaled_scatter_proj_data.get_exam_info_sptr(),
                                   scaled_scatter_proj_data.get_proj_data_info_sptr()->create_shared_clone());
      scatter_normalisation.undo(interpolated_scatter);
      Array<2, float> scale_factors;

      if (min_scale_factor == max_scale_factor)
        {
          if (min_scale_factor == 1.F)
            {
              scaled_scatter_proj_data.fill(interpolated_scatter);
              return; // all done
            }

          // Set all scale_factors to min_scale_factor (which is equal to max_scale_factor here)
          const ProjDataInfo& proj_data_info = *scaled_scatter_proj_data.get_proj_data_info_sptr();
          VectorWithOffset<IndexRange<1>> sinogram_range_vec(proj_data_info.get_min_segment_num(),
                                                             proj_data_info.get_max_segment_num());
          for (int segment_num = proj_data_info.get_min_segment_num(); segment_num <= proj_data_info.get_max_segment_num();
               ++segment_num)
            {
              sinogram_range_vec[segment_num] = IndexRange<1>(proj_data_info.get_min_axial_pos_num(segment_num),
                                                              proj_data_info.get_max_axial_pos_num(segment_num));
            }
          scale_factors.grow(IndexRange2D(sinogram_range_vec));
          scale_factors.fill(min_scale_factor);
        }
      else
        {
          info("upsample_and_fit_scatter_estimate: Finding scale factors by sinogram", 3);
          scale_factors = get_scale_factors_per_sinogram(emission_proj_data, interpolated_scatter, weights_proj_data);

          info(boost::format("upsample_and_fit_scatter_estimate: scale factors before thresholding:\n%1%") % scale_factors, 2);

          threshold_lower(scale_factors.begin_all(), scale_factors.end_all(), min_scale_factor);
          threshold_upper(scale_factors.begin_all(), scale_factors.end_all(), max_scale_factor);
          info(boost::format("upsample_and_fit_scatter_estimate: scale factors after thresholding:\n%1%") % scale_factors, 2);
          VectorWithOffset<float> kernel(-static_cast<int>(half_filter_width), half_filter_width);
          kernel.fill(1.F / (2 * half_filter_width + 1));
          ArrayFilter1DUsingConvolution<float> lowpass_filter(kernel, BoundaryConditions::constant);
          std::for_each(scale_factors.begin(), scale_factors.end(), lowpass_filter);
          info(boost::format("upsample_and_fit_scatter_estimate: scale factors after filtering:\n%1%") % scale_factors, 2);
        }
      info("upsample_and_fit_scatter_estimate: applying scale factors", 3);
      if (scale_sinograms(scaled_scatter_proj_data, interpolated_scatter, scale_factors) != Succeeded::yes)
        {
          error("upsample_and_fit_scatter_estimate: writing of scaled sinograms failed");
        }
    }
  else // min/max_scale_factor equal to 1 and no norm
    {
      inverse_SSRB(scaled_scatter_proj_data, interpolated_direct_scatter);
    }
}

END_NAMESPACE_STIR
