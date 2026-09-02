//
//
/*
    Copyright (C) 2026 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_PSV_PreconditionedGradientReconstruction_h__
#define __stir_PSV_PreconditionedGradientReconstruction_h__
/*!
  \file
  \ingroup PreconditionedGradient
  \brief Declaration of class stir::PreconditionedGradientReconstruction

  \author Manil Sabeur
*/

#include "stir/recon_buildblock/IterativeReconstruction.h"
#include "stir/recon_buildblock/preconditioners/GeneralisedPreconditioner.h"
#include "stir/recon_buildblock/gradient_estimators/GeneralisedGradientEstimator.h"
#include "stir/RegisteredParsingObject.h"
#include <vector>

START_NAMESPACE_STIR

/*!
  \ingroup PreconditionedGradient
  \brief Composable preconditioned-gradient reconstruction engine ("manual" mode).

  The update is a projected, preconditioned gradient step

  \f[ \lambda^{k+1} = \left[ \lambda^k + \alpha_k\, P\, \tilde g_k \right]_+ \f]

  composing two enfichable axes chosen from the \c .par file:
  - <tt>gradient estimator type :=</tt> the estimator that produces \f$\tilde g\f$
    (Full / Ordered Subsets / SVRG / SAGA);
  - <tt>preconditioner type :=</tt> the preconditioner \f$P\f$ (Identity / EM / Hessian Diagonal).

  The step is \f$\alpha_k = s_0/(1+\eta k)\f$; a schedule controls when \f$P\f$ is recomputed.
  If an axis is left unspecified, a Hessian-diagonal preconditioner / SVRG estimator is built with
  its own defaults.

  \note The prior, projector and data model are \em not axes of this engine: they live in the
  objective function (already \c .par-composable). This engine ascends whatever objective it is
  given. Its "auto" counterpart is stir::PreconditionedSVRGReconstruction (registered name \c PSV),
  a locked preset reproducing the PETRIC2 "psv" algorithm.
*/
template <class TargetT>
class PreconditionedGradientReconstruction : public RegisteredParsingObject<PreconditionedGradientReconstruction<TargetT>,
                                                                            Reconstruction<TargetT>,
                                                                            IterativeReconstruction<TargetT>>
{
private:
  typedef RegisteredParsingObject<PreconditionedGradientReconstruction<TargetT>,
                                  Reconstruction<TargetT>,
                                  IterativeReconstruction<TargetT>>
      base_type;

public:
  //! Name used when parsing a Reconstruction object
  static const char* const registered_name;

  PreconditionedGradientReconstruction();
  explicit PreconditionedGradientReconstruction(const std::string& parameter_filename);

  std::string method_info() const override;
  Succeeded set_up(shared_ptr<TargetT> const& target_image_ptr) override;
  void update_estimate(TargetT& current_image_estimate) override;

  //! \name the two composable axes
  //! Equivalent to the \c .par keys \c 'gradient estimator type' and \c 'preconditioner type'.
  //! Both have to be set (there is no default): set_up() refuses otherwise.
  //!@{
  void set_gradient_estimator_sptr(const shared_ptr<GeneralisedGradientEstimator<TargetT>>& arg);
  void set_preconditioner_sptr(const shared_ptr<GeneralisedPreconditioner<TargetT>>& arg);
  const GeneralisedGradientEstimator<TargetT>& get_gradient_estimator() const;
  const GeneralisedPreconditioner<TargetT>& get_preconditioner() const;
  //!@}

  //! \name step size and preconditioner schedule
  //!@{
  //! \f$s_0\f$ in \f$\alpha_k = s_0/(1+\eta k)\f$
  void set_initial_step_size(double arg);
  //! \f$\eta\f$ in \f$\alpha_k = s_0/(1+\eta k)\f$
  void set_step_size_decay(double arg);
  //! if true, \f$k\f$ counts epochs instead of updates (the OSSPS relaxation)
  void set_step_size_decay_per_epoch(bool arg);
  //! recompute \f$P\f$ every \a arg updates (1 = every subiteration, 0 = only the explicit lists)
  void set_precond_update_interval(int arg);
  //! FWHM (mm) of the Gaussian applied to the initial image (0 disables)
  void set_initial_image_filter_fwhm_mm(double arg);
  //!@}

protected:
  //! @name Parameters exposed to the .par file
  //! @{
  double initial_step_size;  //!< \f$s_0\f$
  double step_size_decay;    //!< \f$\eta\f$ in \f$\alpha_k = s_0/(1+\eta k)\f$
  bool step_decay_per_epoch; //!< if true, decay uses the epoch \f$\lfloor n/M\rfloor\f$ (OSSPS relaxation) instead of the update
                             //!< index \f$k\f$
  double pre_filter_fwhm_mm; //!< FWHM (mm) of the Gaussian applied to the initial image
  std::vector<int> precond_update_epochs;        //!< epochs at whose start \f$P\f$ is recomputed
  std::vector<int> precond_update_subiterations; //!< extra update indices at which \f$P\f$ is recomputed
  int precond_update_interval; //!< recompute \f$P\f$ every N updates (1 = every subiteration, as EM/SPS need; 0 = off, only the
                               //!< lists)
  shared_ptr<GeneralisedPreconditioner<TargetT>> preconditioner_sptr;        //!< the preconditioner \f$P\f$
  shared_ptr<GeneralisedGradientEstimator<TargetT>> gradient_estimator_sptr; //!< the gradient estimator
  //! @}

  void set_defaults() override;
  void initialise_keymap() override;

  //! set up the (already chosen or defaulted) preconditioner + estimator, the initial filter and P.
  //! Shared by this engine and its presets (called after the objective is set up).
  Succeeded set_up_components(shared_ptr<TargetT> const& target_image_ptr);

private:
  shared_ptr<TargetT> precond_sptr; //!< current preconditioner image \f$P\f$
  bool precond_computed = false;    //!< whether \f$P\f$ has been computed at least once
};

END_NAMESPACE_STIR

#endif
