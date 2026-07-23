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
  \brief Declaration of class stir::PreconditionedSVRGReconstruction

  \author Manil Sabeur

  \par Attribution

  This is a native C++ readaptation of the "psv" (preconditioned SVRG) contribution to the
  PETRIC2 reconstruction challenge, by:
  - Hok Shing Wong, University of Bath, UK
  - Margaret Duff, STFC, UK
  - Matthias Ehrhardt, University of Bath, UK
  - Georg Schramm, KU Leuven, Belgium

  The original (PETRIC2-Speedy, branch \c psv, https://github.com/SyneRBI/PETRIC2-Speedy/tree/psv)
  is pure Python running on SIRF+CIL; the algorithm itself is the same as "ALG1" of the MaGeZ
  contribution to PETRIC1. This class reimplements that algorithm on STIR primitives, driven by a
  \c .par file, with no SIRF and no CIL. All credit for the algorithm and its hyper-parameters
  belongs to the original authors; this file only ports it to native STIR.

  \par References

  - M. J. Ehrhardt et al., "Fast PET reconstruction with variance reduction and prior-aware
    preconditioning," Front. Nucl. Med., 2025. https://doi.org/10.3389/fnume.2025.1641215
  - R. Twyman et al., "An Investigation of Stochastic Variance Reduction Algorithms for Relative
    Difference Penalized 3D PET Image Reconstruction," IEEE TMI, 2023.
  - J. Nuyts et al., "A concave prior penalizing relative differences for maximum-a-posteriori
    reconstruction in emission tomography," IEEE TNS, vol. 49, 2022.

  \par Validation

  This class mirrors a validated Python prototype
  (Env/petric2_speedy/src/psv_proto/preconditioned_svrg.py), whose behaviour was established by
  validations val01–val07. The prototype is the oracle; this class is validated against it by
  EQUIVALENCE (does it meet the PETRIC criterion in a comparable number of epochs), not by
  trajectory — see the class documentation.
*/

#ifndef __stir_PSV_PreconditionedSVRGReconstruction_h__
#define __stir_PSV_PreconditionedSVRGReconstruction_h__

#include "stir/recon_buildblock/IterativeReconstruction.h"
#include "stir/DataProcessor.h"
#include "stir/RegisteredParsingObject.h"
#include <vector>

START_NAMESPACE_STIR

/*!
  \ingroup PSV
  \brief Preconditioned Stochastic Variance Reduced Gradient (SVRG) reconstruction.

  Native STIR implementation of the PETRIC2 "psv" algorithm. The update is

  \f[ \lambda^{k+1} = \left[ \lambda^k + \alpha_k\, P\, \tilde g_k \right]_+ \f]

  where the projection \f$[\cdot]_+\f$ enforces non-negativity, and

  - \f$\tilde g_k = N\left(\nabla f_i(\lambda^k) - \nabla f_i(\tilde\lambda)\right)
    + \sum_j \nabla f_j(\tilde\lambda)\f$ is the SVRG variance-reduced gradient
    estimate, with \f$\tilde\lambda\f$ a periodically-refreshed snapshot at which all
    subset gradients have been computed;
  - \f$P = m\, \dfrac{\lambda + \delta\,\max(\lambda_s)}
    {A^\mathrm{T}1 + c\, H_{jj}\, \lambda}\f$ is a diagonal preconditioner, with \f$m\f$
    the FOV mask, \f$A^\mathrm{T}1\f$ the sensitivity (sum of subset sensitivities),
    \f$H_{jj}\f$ the diagonal of the Hessian of the prior evaluated on a Gaussian-smoothed
    image, \f$c\f$ = \c precond_hessian_factor, and \f$\delta\f$ = \c precond_delta_rel;
  - \f$\alpha_k = s_0 / (1 + \eta k)\f$ the decreasing step size, \f$k\f$ the update
    (subiteration) index.

  \par Mapping onto the STIR subiteration loop

  IterativeReconstruction::reconstruct() calls update_estimate() once per subiteration.
  Here \c subiteration_num is the global update counter \f$k\f$, and the current epoch is
  \f$(\mathtt{subiteration\_num}-1)/\mathtt{num\_subsets}\f$. The subset for each
  subiteration comes from the base class (get_subset_num()); the SVRG estimator is
  unbiased for any subset order (verified in validation val03), so the base class's own
  ordering is used rather than reproducing the prototype's numpy shuffle.

  \par Validation contract

  This class is validated against the Python prototype by EQUIVALENCE, not trajectory:
  the numpy and C++ subset orders cannot coincide, and the project target is "equivalent
  quality, not bit-exact reproduction". The check is: does the reconstruction meet the
  PETRIC thresholds in a comparable number of epochs on the same machine?

  \par Preconditioner-specific traps (each established by a validation — do not "fix")

  - The prior gradient is already divided by \c num_subsets inside
    GeneralisedObjectiveFunction::compute_sub_gradient. The penalisation factor must
    therefore stay whole; do NOT additionally divide it (val03).
  - \c precond_hessian_factor (the "2.0" of psv) is absent from the psv README but present
    in its code. It weights only the prior term of the denominator, so it rebalances prior
    against likelihood voxel-wise and is not absorbable by \c initial_step_size.
  - \f$H_{jj}\f$ is evaluated on the Gaussian-SMOOTHED image, while the \f$\lambda\f$
    multiplying it in the denominator is NOT smoothed.
  - Without \c precond_delta_rel, \f$P\f$ vanishes wherever \f$\lambda = 0\f$, so any voxel
    clamped to zero freezes forever (val04). psv defaults it to 0; a small value unfreezes.
  - The FOV mask is essential, not cosmetic: outside the FOV \f$A^\mathrm{T}1\to 0\f$, and
    without the mask \f$P\f$ would explode there (val06 / prototype).

  \warning This class should be the last in the Reconstruction hierarchy.
*/
template <class TargetT>
class PreconditionedSVRGReconstruction : public RegisteredParsingObject<PreconditionedSVRGReconstruction<TargetT>,
                                                                        Reconstruction<TargetT>,
                                                                        IterativeReconstruction<TargetT>>
{
private:
  typedef RegisteredParsingObject<PreconditionedSVRGReconstruction<TargetT>,
                                  Reconstruction<TargetT>,
                                  IterativeReconstruction<TargetT>>
      base_type;

public:
  //! Name used when parsing a Reconstruction object
  static const char* const registered_name;

  //! Default constructor (calls set_defaults())
  PreconditionedSVRGReconstruction();

  //! Construct from a parameter file (or ask_parameters() when filename == "")
  explicit PreconditionedSVRGReconstruction(const std::string& parameter_filename);

  //! gives method information
  std::string method_info() const override;

  //! operations prior to the iterations (allocates SVRG state, A^T 1, FOV mask, filter)
  Succeeded set_up(shared_ptr<TargetT> const& target_image_ptr) override;

  //! one preconditioned SVRG update — the principal per-subiteration operation
  void update_estimate(TargetT& current_image_estimate) override;

protected:
  //! @name Parameters exposed to the .par file
  //! @{

  //! initial step size \f$s_0\f$
  double initial_step_size;
  //! step-size decay \f$\eta\f$ in \f$\alpha_k = s_0/(1+\eta k)\f$
  double step_size_decay;
  //! Hessian factor \f$c\f$ in the preconditioner denominator (psv default 2.0)
  double precond_hessian_factor;
  //! FWHM (mm) of the Gaussian applied before computing \f$H_{jj}\f$
  double precond_filter_fwhm_mm;
  //! FWHM (mm) of the Gaussian applied to the initial image
  double pre_filter_fwhm_mm;
  //! small relative number keeping \f$P>0\f$ where \f$\lambda=0\f$ (psv default 0)
  double precond_delta_rel;
  //! recompute all subset gradients (SVRG snapshot) every this many epochs
  /*! psv uses epochs {0,2,4,...}, i.e. an interval of 2. */
  int complete_gradient_epoch_interval;
  //! epochs at whose START the preconditioner is recomputed (psv default {1})
  /*! The preconditioner is ALWAYS computed once before the first update (init). */
  std::vector<int> precond_update_epochs;
  //! extra global update indices at which to recompute the preconditioner (psv default {4})
  /*! This expresses psv's "after the 4th update" — a mid-epoch, update-level event that an
      epoch interval cannot represent (main.py:365, "or self._update == 4"). */
  std::vector<int> precond_update_subiterations;
  //! floor added to \f$A^\mathrm{T}1\f$ to avoid division by zero
  double adjoint_ones_floor;
  //! optional file with a 0/1 FOV mask; if empty, derived by thresholding \f$A^\mathrm{T}1\f$
  std::string fov_mask_filename;
  //! threshold (relative to max \f$A^\mathrm{T}1\f$) used to derive the FOV mask
  double fov_threshold;

  //! @}

  void set_defaults() override;
  void initialise_keymap() override;
  bool post_processing() override;

private:
  //! @name SVRG state, allocated in set_up()
  //! @{
  shared_ptr<TargetT> adjoint_ones_sptr;                  //!< \f$A^\mathrm{T}1\f$
  shared_ptr<TargetT> precond_sptr;                       //!< current preconditioner \f$P\f$
  shared_ptr<TargetT> summed_subset_gradients_sptr;       //!< \f$\sum_j \nabla f_j(\tilde\lambda)\f$
  std::vector<shared_ptr<TargetT>> subset_gradients;      //!< snapshot gradients \f$\nabla f_i(\tilde\lambda)\f$
  shared_ptr<TargetT> fov_mask_sptr;                      //!< 0/1 mask on the FOV
  shared_ptr<DataProcessor<TargetT>> precond_filter_sptr; //!< Gaussian filter for \f$H_{jj}\f$
  //! @}

  //! (re)compute the preconditioner \f$P\f$ from the current image
  void compute_preconditioner(const TargetT& current_image_estimate);
  //! recompute all subset gradients and their sum (the SVRG snapshot)
  void update_all_subset_gradients(const TargetT& current_image_estimate);
};

END_NAMESPACE_STIR

#endif
