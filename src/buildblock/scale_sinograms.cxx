//
//
/*
  Copyright (C) 2004- 2009, Hammersmith Imanet Ltd
  Copyright (C) 2019, University College London
  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup projdata
  \brief Implementations of functions defined in scale_sinogram.h

  \author Charalampos Tsoumpas
  \author Kris Thielemans

*/
#include "stir/scale_sinograms.h"
#include "stir/ProjData.h"
#include "stir/ProjDataInfo.h"
#include "stir/Bin.h"
#include "stir/Sinogram.h"
#include "stir/Succeeded.h"
#include "stir/warning.h"

START_NAMESPACE_STIR

Succeeded
scale_sinograms(ProjData& scaled_scatter_proj_data, const ProjData& scatter_proj_data, const Array<2, float> scale_factors)
{
  const ProjDataInfo& proj_data_info = dynamic_cast<const ProjDataInfo&>(*scaled_scatter_proj_data.get_proj_data_info_sptr());

  Bin bin;

  for (bin.segment_num() = proj_data_info.get_min_segment_num(); bin.segment_num() <= proj_data_info.get_max_segment_num();
       ++bin.segment_num())
    for (bin.axial_pos_num() = proj_data_info.get_min_axial_pos_num(bin.segment_num());
         bin.axial_pos_num() <= proj_data_info.get_max_axial_pos_num(bin.segment_num());
         ++bin.axial_pos_num())
      {
        // -- UPSAMPLE FIX 3 --
        // Most of the fix are the same here, we just add a loop over the TOF bins
        // and we adapt the code accordingly
        for (int timing_pos_num = proj_data_info.get_min_tof_pos_num();
             timing_pos_num <= proj_data_info.get_max_tof_pos_num();
             ++timing_pos_num)
          {
            Sinogram<float> scatter_sinogram = scatter_proj_data.get_sinogram(bin.axial_pos_num(), bin.segment_num(), false, timing_pos_num);
            Sinogram<float> scaled_sinogram = scatter_sinogram;
            scaled_sinogram *= scale_factors[bin.segment_num()][bin.axial_pos_num()];

            if (scaled_scatter_proj_data.set_sinogram(scaled_sinogram) == Succeeded::no)
              return Succeeded::no;
          }
      }
  return Succeeded::yes;
}

Array<2, float>
get_scale_factors_per_sinogram(const ProjData& numerator_proj_data,
                               const ProjData& denominator_proj_data,
                               const ProjData& weights_proj_data)
{

  const ProjDataInfo& proj_data_info = dynamic_cast<const ProjDataInfo&>(*weights_proj_data.get_proj_data_info_sptr());

  Bin bin;

  // scale factor to use when the denominator is zero
  const float default_scale = 1.F;

  // Build sinogram_range via VectorWithOffset constructor so that is_regular_range
  // is set to regular_to_do (not regular_true). This forces size_all() to sum
  // each segment's axial count rather than using the fast-path 17*1=17, which
  // would cause a heap-buffer-overflow when segments have different axial sizes.
  VectorWithOffset<IndexRange<1>> sinogram_range_vec(proj_data_info.get_min_segment_num(),
                                                     proj_data_info.get_max_segment_num());
  for (int segment_num = proj_data_info.get_min_segment_num(); segment_num <= proj_data_info.get_max_segment_num(); ++segment_num)
    {
      sinogram_range_vec[segment_num] = IndexRange<1>(proj_data_info.get_min_axial_pos_num(segment_num),
                                                      proj_data_info.get_max_axial_pos_num(segment_num));
    }
  IndexRange2D sinogram_range(sinogram_range_vec);
  Array<2, float> total_in_denominator(sinogram_range);
  Array<2, float> total_in_numerator(sinogram_range);
  Array<2, float> scale_factors(sinogram_range);

  for (bin.segment_num() = proj_data_info.get_min_segment_num(); bin.segment_num() <= proj_data_info.get_max_segment_num();
       ++bin.segment_num())
    for (bin.axial_pos_num() = proj_data_info.get_min_axial_pos_num(bin.segment_num());
         bin.axial_pos_num() <= proj_data_info.get_max_axial_pos_num(bin.segment_num());
         ++bin.axial_pos_num())
      {
        // Use weights from TOF bin 0 (the tail mask is identical across TOF bins).
        // Sum numerator and denominator over all TOF bins so that the scale factor
        // is estimated from the full statistics rather than a single TOF bin.
        const Sinogram<float> weights = weights_proj_data.get_sinogram(bin.axial_pos_num(), bin.segment_num());

        float denom_total_sum = 0.F;
        total_in_denominator[bin.segment_num()][bin.axial_pos_num()] = 0.F;
        total_in_numerator[bin.segment_num()][bin.axial_pos_num()] = 0.F;

        for (int timing_pos_num = proj_data_info.get_min_tof_pos_num();
             timing_pos_num <= proj_data_info.get_max_tof_pos_num();
             ++timing_pos_num)
          {
            const Sinogram<float> denom_sino
                = denominator_proj_data.get_sinogram(bin.axial_pos_num(), bin.segment_num(), false, timing_pos_num);
            const Sinogram<float> num_sino
                = numerator_proj_data.get_sinogram(bin.axial_pos_num(), bin.segment_num(), false, timing_pos_num);
            total_in_denominator[bin.segment_num()][bin.axial_pos_num()]
                += (denom_sino * weights).sum();
            total_in_numerator[bin.segment_num()][bin.axial_pos_num()]
                += (num_sino * weights).sum();
            denom_total_sum += denom_sino.sum();
          }

        if (denom_total_sum == 0.f)
          {
            scale_factors[bin.segment_num()][bin.axial_pos_num()] = default_scale;
          }
        else
          {
            if (total_in_denominator[bin.segment_num()][bin.axial_pos_num()]
                <= denom_total_sum
                       / (proj_data_info.get_num_views() * proj_data_info.get_num_tangential_poss())
                       * .001f)
              {
                warning("Problem at segment %d, axial pos %d in finding sinogram scaling factor.\n"
                        "Weighted data in denominator %g is very small compared to total in sinogram %g.\n"
                        "Adjust weights?.\n"
                        "I will use scale factor %g",
                        bin.segment_num(),
                        bin.axial_pos_num(),
                        total_in_denominator[bin.segment_num()][bin.axial_pos_num()],
                        denom_total_sum,
                        default_scale);
                scale_factors[bin.segment_num()][bin.axial_pos_num()] = default_scale;
              }
            else
              {
                scale_factors[bin.segment_num()][bin.axial_pos_num()]
                    = total_in_numerator[bin.segment_num()][bin.axial_pos_num()]
                      / total_in_denominator[bin.segment_num()][bin.axial_pos_num()];
              }
          }
      }

  return scale_factors;
}

END_NAMESPACE_STIR
