//
//
/*
    Copyright (C) 2026 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
\file
\ingroup PSV
\ingroup reconstructors
\brief  implementation of the stir::PreconditionedSVRGReconstruction class

\author Manil Sabeur

  Native C++ readaptation of the PETRIC2 "psv" contribution by Hok Shing Wong (Univ. of Bath),
  Margaret Duff (STFC), Matthias Ehrhardt (Univ. of Bath) and Georg Schramm (KU Leuven).
  See the class header for the full attribution and references.
*/

#include "stir/PSV/PreconditionedSVRGReconstruction.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMean.h"
#include "stir/recon_buildblock/GeneralisedPrior.h"
#include "stir/SeparableGaussianImageFilter.h"
#include "stir/Succeeded.h"
#include "stir/thresholding.h"
#include "stir/is_null_ptr.h"
#include "stir/IO/read_from_file.h"
#include "stir/info.h"
#include "stir/warning.h"
#include "stir/error.h"
#include "stir/format.h"

#include <memory>
#include <algorithm>
#include <sstream>
#include "boost/lambda/lambda.hpp"
#include "stir/unique_ptr.h"

using boost::lambda::_1;
using boost::lambda::_2;

START_NAMESPACE_STIR

//*************** parameters *************

template <typename TargetT>
const char* const PreconditionedSVRGReconstruction<TargetT>::registered_name = "PSV";

template <class TargetT>
void
PreconditionedSVRGReconstruction<TargetT>::set_defaults()
{
  base_type::set_defaults();

  // defaults reproduce psv (PETRIC2-Speedy, branch psv). Tuning is done through .par overrides,
  // never by changing these.
  this->initial_step_size = 3.5;
  this->step_size_decay = 0.01;
  this->precond_hessian_factor = 2.0;
  this->precond_filter_fwhm_mm = 5.0;
  this->pre_filter_fwhm_mm = 6.0;
  this->precond_delta_rel = 0.0;
  this->complete_gradient_epoch_interval = 2;
  this->precond_update_epochs = std::vector<int>(1, 1);
  this->precond_update_subiterations = std::vector<int>(1, 4);
  this->adjoint_ones_floor = 1e-6;
  this->fov_mask_filename = "";
  this->fov_threshold = 1e-3;
}

template <class TargetT>
void
PreconditionedSVRGReconstruction<TargetT>::initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("PSVParameters");
  this->parser.add_stop_key("End");

  this->parser.add_key("initial step size", &this->initial_step_size);
  this->parser.add_key("step size decay", &this->step_size_decay);
  this->parser.add_key("precond hessian factor", &this->precond_hessian_factor);
  this->parser.add_key("precond filter fwhm (mm)", &this->precond_filter_fwhm_mm);
  this->parser.add_key("initial image filter fwhm (mm)", &this->pre_filter_fwhm_mm);
  this->parser.add_key("precond delta rel", &this->precond_delta_rel);
  this->parser.add_key("complete gradient epoch interval", &this->complete_gradient_epoch_interval);
  this->parser.add_key("precond update epochs", &this->precond_update_epochs);
  this->parser.add_key("precond update subiterations", &this->precond_update_subiterations);
  this->parser.add_key("adjoint ones floor", &this->adjoint_ones_floor);
  this->parser.add_key("FOV mask filename", &this->fov_mask_filename);
  this->parser.add_key("FOV threshold", &this->fov_threshold);
}

template <class TargetT>
bool
PreconditionedSVRGReconstruction<TargetT>::post_processing()
{
  if (base_type::post_processing())
    return true;

  if (this->initial_step_size <= 0)
    {
      warning("PSV: initial step size should be strictly positive but is %g", this->initial_step_size);
      return true;
    }
  if (this->step_size_decay < 0)
    {
      warning("PSV: step size decay should be non-negative but is %g", this->step_size_decay);
      return true;
    }
  if (this->precond_hessian_factor < 0)
    {
      warning("PSV: precond hessian factor should be non-negative but is %g", this->precond_hessian_factor);
      return true;
    }
  if (this->complete_gradient_epoch_interval < 1)
    {
      warning("PSV: complete gradient epoch interval should be at least 1 but is %d", this->complete_gradient_epoch_interval);
      return true;
    }
  return false;
}

//*************** other functions *************

template <class TargetT>
PreconditionedSVRGReconstruction<TargetT>::PreconditionedSVRGReconstruction()
{
  set_defaults();
}

template <class TargetT>
PreconditionedSVRGReconstruction<TargetT>::PreconditionedSVRGReconstruction(const std::string& parameter_filename)
{
  this->initialise(parameter_filename);
  info(this->parameter_info());
}

template <class TargetT>
std::string
PreconditionedSVRGReconstruction<TargetT>::method_info() const
{
  std::ostringstream s;

  if (!this->objective_function_sptr->prior_is_zero())
    s << "MAP-";
  if (this->num_subsets > 1)
    s << "OS-";
  s << "preconditioned SVRG";

  return s.str();
}

template <class TargetT>
Succeeded
PreconditionedSVRGReconstruction<TargetT>::set_up(shared_ptr<TargetT> const& target_image_ptr)
{
  if (base_type::set_up(target_image_ptr) == Succeeded::no)
    return Succeeded::no;

  // The preconditioner needs A^T 1 (the sensitivity), which is Poisson-specific. This class is
  // therefore not generic over all objective functions: require a Poisson log-likelihood.
  PoissonLogLikelihoodWithLinearModelForMean<TargetT>* const poisson_obj_ptr
      = dynamic_cast<PoissonLogLikelihoodWithLinearModelForMean<TargetT>*>(this->objective_function_sptr.get());
  if (is_null_ptr(poisson_obj_ptr))
    {
      warning("PSV: objective function must be of a type derived from "
              "PoissonLogLikelihoodWithLinearModelForMean (the preconditioner needs the subset sensitivities)\n");
      return Succeeded::no;
    }

  const int num_subsets = this->get_num_subsets();

  // A^T 1 = sum of the subset sensitivities (this identity is verified in validation val03)
  this->adjoint_ones_sptr.reset(target_image_ptr->get_empty_copy());
  std::fill(this->adjoint_ones_sptr->begin_all(), this->adjoint_ones_sptr->end_all(), 0.F);
  for (int subset_num = 0; subset_num < num_subsets; ++subset_num)
    std::transform(this->adjoint_ones_sptr->begin_all(),
                   this->adjoint_ones_sptr->end_all(),
                   poisson_obj_ptr->get_subset_sensitivity(subset_num).begin_all_const(),
                   this->adjoint_ones_sptr->begin_all(),
                   _1 + _2);

  // FOV mask, from file or by thresholding A^T 1. This is essential, not cosmetic: outside the FOV
  // A^T 1 -> 0, and without the mask the preconditioner would explode there.
  this->fov_mask_sptr.reset(target_image_ptr->get_empty_copy());
  if (this->fov_mask_filename != "")
    {
      shared_ptr<TargetT> mask_sptr = read_from_file<TargetT>(this->fov_mask_filename);
      std::string explanation;
      if (!mask_sptr->has_same_characteristics(*target_image_ptr, explanation))
        {
          warning("PSV: FOV mask should have the same characteristics as the target image: %s", explanation.c_str());
          return Succeeded::no;
        }
      *this->fov_mask_sptr = *mask_sptr;
    }
  else
    {
      const float threshold = static_cast<float>(this->fov_threshold)
                              * (*std::max_element(this->adjoint_ones_sptr->begin_all(), this->adjoint_ones_sptr->end_all()));
      std::transform(this->adjoint_ones_sptr->begin_all_const(),
                     this->adjoint_ones_sptr->end_all_const(),
                     this->fov_mask_sptr->begin_all(),
                     [threshold](float a) { return a > threshold ? 1.F : 0.F; });
    }

  // floor A^T 1 to avoid division by zero
  threshold_min_to_small_positive_value(
      this->adjoint_ones_sptr->begin_all(), this->adjoint_ones_sptr->end_all(), static_cast<float>(this->adjoint_ones_floor));

  // Gaussian filter used before evaluating the Hessian diagonal
  {
    shared_ptr<SeparableGaussianImageFilter<float>> filter_sptr(new SeparableGaussianImageFilter<float>);
    BasicCoordinate<3, float> fwhms;
    fwhms[1] = fwhms[2] = fwhms[3] = static_cast<float>(this->precond_filter_fwhm_mm);
    filter_sptr->set_fwhms(fwhms);
    if (filter_sptr->set_up(*target_image_ptr) == Succeeded::no)
      {
        warning("PSV: could not set up the preconditioner Gaussian filter\n");
        return Succeeded::no;
      }
    this->precond_filter_sptr = filter_sptr;
  }

  // post-filter the initial (OSEM) image with an isotropic Gaussian (psv uses 6 mm FWHM)
  if (this->pre_filter_fwhm_mm > 0)
    {
      SeparableGaussianImageFilter<float> pre_filter;
      BasicCoordinate<3, float> fwhms;
      fwhms[1] = fwhms[2] = fwhms[3] = static_cast<float>(this->pre_filter_fwhm_mm);
      pre_filter.set_fwhms(fwhms);
      if (pre_filter.set_up(*target_image_ptr) == Succeeded::yes)
        pre_filter.apply(*target_image_ptr);
    }

  // allocate the SVRG state
  this->subset_gradients.resize(num_subsets);
  for (int subset_num = 0; subset_num < num_subsets; ++subset_num)
    this->subset_gradients[subset_num].reset(target_image_ptr->get_empty_copy());
  this->summed_subset_gradients_sptr.reset(target_image_ptr->get_empty_copy());
  this->precond_sptr.reset(); // forces computation before the very first update

  return Succeeded::yes;
}

template <class TargetT>
void
PreconditionedSVRGReconstruction<TargetT>::compute_preconditioner(const TargetT& current_image_estimate)
{
  info("PSV: recomputing preconditioner");

  // numerator = mask * (x + delta_rel * max(x_smoothed))
  unique_ptr<TargetT> numerator_ptr(current_image_estimate.clone());
  if (this->precond_delta_rel != 0)
    {
      unique_ptr<TargetT> smoothed_ptr(current_image_estimate.clone());
      this->precond_filter_sptr->apply(*smoothed_ptr);
      const float bump
          = static_cast<float>(this->precond_delta_rel) * (*std::max_element(smoothed_ptr->begin_all(), smoothed_ptr->end_all()));
      std::transform(numerator_ptr->begin_all(), numerator_ptr->end_all(), numerator_ptr->begin_all(), _1 + bump);
    }
  std::transform(numerator_ptr->begin_all(),
                 numerator_ptr->end_all(),
                 this->fov_mask_sptr->begin_all(),
                 numerator_ptr->begin_all(),
                 _1 * _2);

  // denominator = A^T 1 + c * H_jj(x_smoothed) * x
  unique_ptr<TargetT> denominator_ptr(current_image_estimate.get_empty_copy());
  if (this->objective_function_sptr->prior_is_zero())
    {
      // H_jj = 0 without a prior: the preconditioner reduces to the MLEM one
      *denominator_ptr = *this->adjoint_ones_sptr;
    }
  else
    {
      unique_ptr<TargetT> smoothed_ptr(current_image_estimate.clone());
      this->precond_filter_sptr->apply(*smoothed_ptr); // H_jj is evaluated on the SMOOTHED image
      this->objective_function_sptr->get_prior_ptr()->compute_Hessian_diagonal(*denominator_ptr, *smoothed_ptr);
      const float factor = static_cast<float>(this->precond_hessian_factor);
      // c * H_jj * x, then + A^T 1. Note x is the UN-smoothed image.
      std::transform(denominator_ptr->begin_all(),
                     denominator_ptr->end_all(),
                     current_image_estimate.begin_all_const(),
                     denominator_ptr->begin_all(),
                     _1 * _2);
      std::transform(denominator_ptr->begin_all(),
                     denominator_ptr->end_all(),
                     this->adjoint_ones_sptr->begin_all(),
                     denominator_ptr->begin_all(),
                     _1 * factor + _2);
    }
  threshold_min_to_small_positive_value(
      denominator_ptr->begin_all(), denominator_ptr->end_all(), static_cast<float>(this->adjoint_ones_floor));

  if (is_null_ptr(this->precond_sptr))
    this->precond_sptr.reset(current_image_estimate.get_empty_copy());
  std::transform(numerator_ptr->begin_all(),
                 numerator_ptr->end_all(),
                 denominator_ptr->begin_all(),
                 this->precond_sptr->begin_all(),
                 _1 / _2);
}

template <class TargetT>
void
PreconditionedSVRGReconstruction<TargetT>::update_all_subset_gradients(const TargetT& current_image_estimate)
{
  info("PSV: recomputing all subset gradients (SVRG snapshot)");
  const int num_subsets = this->get_num_subsets();
  std::fill(this->summed_subset_gradients_sptr->begin_all(), this->summed_subset_gradients_sptr->end_all(), 0.F);
  for (int subset_num = 0; subset_num < num_subsets; ++subset_num)
    {
      this->objective_function_sptr->compute_sub_gradient(
          *this->subset_gradients[subset_num], current_image_estimate, subset_num);
      std::transform(this->summed_subset_gradients_sptr->begin_all(),
                     this->summed_subset_gradients_sptr->end_all(),
                     this->subset_gradients[subset_num]->begin_all(),
                     this->summed_subset_gradients_sptr->begin_all(),
                     _1 + _2);
    }
}

/*! \brief preconditioned SVRG additive update at every subiteration
  \warning This modifies the stored SVRG snapshot gradients. So you <strong>have to</strong>
  call set_up() before running a new reconstruction.
  */
template <class TargetT>
void
PreconditionedSVRGReconstruction<TargetT>::update_estimate(TargetT& current_image_estimate)
{
  this->check(current_image_estimate);

  const int num_subsets = this->get_num_subsets();
  // update counter, matching psv's global counter (starts at 0)
  const int update_num = this->get_subiteration_num() - this->get_start_subiteration_num();
  const int epoch = update_num / num_subsets;
  const bool at_epoch_start = (update_num % num_subsets == 0);

  // preconditioner schedule: before the very first update, at the start of the listed epochs,
  // and at the listed update indices (psv: init + epoch 1 + update 4)
  const bool at_precond_epoch = at_epoch_start
                                && std::find(this->precond_update_epochs.begin(), this->precond_update_epochs.end(), epoch)
                                       != this->precond_update_epochs.end();
  const bool at_precond_update
      = std::find(this->precond_update_subiterations.begin(), this->precond_update_subiterations.end(), update_num)
        != this->precond_update_subiterations.end();
  if (is_null_ptr(this->precond_sptr) || at_precond_epoch || at_precond_update)
    this->compute_preconditioner(current_image_estimate);

  // gradient: full SVRG snapshot, or the variance-reduced estimate
  unique_ptr<TargetT> gradient_ptr(current_image_estimate.get_empty_copy());
  const bool at_snapshot = at_epoch_start && (epoch % this->complete_gradient_epoch_interval == 0);
  if (at_snapshot)
    {
      this->update_all_subset_gradients(current_image_estimate);
      *gradient_ptr = *this->summed_subset_gradients_sptr;
    }
  else
    {
      const int subset_num = this->get_subset_num();
      info(format("Now processing subset #: {}", subset_num));
      // g_tilde = num_subsets * (grad_i(x) - grad_i(x_snapshot)) + sum_j grad_j(x_snapshot)
      this->objective_function_sptr->compute_sub_gradient(*gradient_ptr, current_image_estimate, subset_num);
      std::transform(gradient_ptr->begin_all(),
                     gradient_ptr->end_all(),
                     this->subset_gradients[subset_num]->begin_all(),
                     gradient_ptr->begin_all(),
                     _1 - _2);
      std::transform(gradient_ptr->begin_all(),
                     gradient_ptr->end_all(),
                     this->summed_subset_gradients_sptr->begin_all(),
                     gradient_ptr->begin_all(),
                     _1 * num_subsets + _2);
    }

  // decreasing step, preconditioned ascent
  const float step_size = static_cast<float>(this->initial_step_size / (1. + this->step_size_decay * update_num));
  info(format("step size = {}", step_size));
  std::transform(gradient_ptr->begin_all(),
                 gradient_ptr->end_all(),
                 this->precond_sptr->begin_all(),
                 gradient_ptr->begin_all(),
                 _1 * _2 * step_size);
  current_image_estimate += *gradient_ptr;

  // enforce the non-negativity constraint
  std::for_each(current_image_estimate.begin_all(), current_image_estimate.end_all(), [](float& a) {
    if (a < 0.F)
      a = 0.F;
  });
}

END_NAMESPACE_STIR

///////// instantiations
#include "stir/DiscretisedDensity.h"
START_NAMESPACE_STIR

template class PreconditionedSVRGReconstruction<DiscretisedDensity<3, float>>;

END_NAMESPACE_STIR
