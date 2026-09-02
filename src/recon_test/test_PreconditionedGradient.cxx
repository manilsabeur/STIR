/*
    Copyright (C) 2026, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup recon_test
  \ingroup PreconditionedGradient
  \brief Test program for stir::PreconditionedGradientReconstruction

  Checks the two things that a composable engine has to get right: that it refuses compositions
  which make no sense (the scale and prior-capability contracts, verified at set_up()), and that
  the compositions which reproduce an existing algorithm actually do so. The latter is tested
  against stir::OSMAPOSLReconstruction without a prior, i.e. against OSEM/MLEM.

  \author Manil Sabeur
*/

#include "stir/recon_buildblock/test/PoissonLLReconstructionTests.h"
#include "stir/PreconditionedGradient/PreconditionedGradientReconstruction.h"
#include "stir/OSMAPOSL/OSMAPOSLReconstruction.h"
#include "stir/recon_buildblock/preconditioners/EMPreconditioner.h"
#include "stir/recon_buildblock/preconditioners/HessianDiagonalPreconditioner.h"
#include "stir/recon_buildblock/preconditioners/SPSPreconditioner.h"
#include "stir/recon_buildblock/gradient_estimators/FullGradientEstimator.h"
#include "stir/recon_buildblock/gradient_estimators/OrderedSubsetsGradientEstimator.h"
#include "stir/recon_buildblock/QuadraticPrior.h"
#include "stir/recon_buildblock/GibbsRelativeDifferencePenalty.h"
#include "stir/Verbosity.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

typedef DiscretisedDensity<3, float> target_type;

/*!
  \ingroup recon_test
  \ingroup PreconditionedGradient
  \brief Test class for PreconditionedGradientReconstruction
*/
class TestPreconditionedGradient : public PoissonLLReconstructionTests<target_type>
{
private:
  typedef PoissonLLReconstructionTests<target_type> base_type;

public:
  TestPreconditionedGradient(const std::string& projector_pair_filename = "",
                             const std::string& proj_data_filename = "",
                             const std::string& density_filename = "")
      : base_type(projector_pair_filename, proj_data_filename, density_filename)
  {}
  ~TestPreconditionedGradient() override {}

  void construct_reconstructor() override;
  PreconditionedGradientReconstruction<target_type>& recon()
  {
    return dynamic_cast<PreconditionedGradientReconstruction<target_type>&>(*this->_recon_sptr);
  }

  void run_tests() override;

private:
  //! builds an engine on a freshly set-up log-likelihood, with \a prior_sptr (possibly null)
  shared_ptr<PreconditionedGradientReconstruction<target_type>>
  construct_engine(const shared_ptr<GeneralisedPrior<target_type>>& prior_sptr, int num_subsets);

  //! set_up() has to refuse this composition
  void check_refused(const std::string& what,
                     const shared_ptr<GeneralisedGradientEstimator<target_type>>& estimator_sptr,
                     const shared_ptr<GeneralisedPreconditioner<target_type>>& preconditioner_sptr,
                     const shared_ptr<GeneralisedPrior<target_type>>& prior_sptr);

  void test_contracts();
  //! runs the composition against OSMAPOSL with \a num_subsets subsets, and compares both
  void test_against_OSMAPOSL(const std::string& what, int num_subsets, bool use_subset_sensitivity);

  //! \name comparisons restricted to the field of view
  /*! The EM preconditioner is zero outside the FOV (where \f$A^T1\f$ is negligible), so the engine
    leaves the initial image untouched there, by design. A whole-image comparison would therefore
    measure that mask rather than the reconstruction, which is why every comparison below is
    restricted to the FOV, exactly as the preconditioner defines it. */
  //!@{
  //! largest \f$|a-b|\f$ inside the FOV, relative to the maximum of \a b
  double max_relative_difference_in_fov(const target_type& a, const target_type& b) const;
  //! mean \f$|a-b|\f$ inside the FOV, relative to the maximum of \a b
  double mean_relative_difference_in_fov(const target_type& a, const target_type& b) const;
  //!@}
};

shared_ptr<PreconditionedGradientReconstruction<target_type>>
TestPreconditionedGradient::construct_engine(const shared_ptr<GeneralisedPrior<target_type>>& prior_sptr, int num_subsets)
{
  this->construct_log_likelihood();
  this->_objective_function_sptr->set_prior_sptr(prior_sptr);
  shared_ptr<PreconditionedGradientReconstruction<target_type>> engine_sptr(
      new PreconditionedGradientReconstruction<target_type>);
  engine_sptr->set_objective_function_sptr(this->_objective_function_sptr);
  engine_sptr->set_num_subsets(num_subsets);
  engine_sptr->set_num_subiterations(num_subsets);
  engine_sptr->set_input_data(this->_proj_data_sptr);
  engine_sptr->set_disable_output(true);
  engine_sptr->set_output_filename_prefix("test_precond_gradient");
  // P depends on the current image for both EM and SPS, so it has to be recomputed every update
  engine_sptr->set_precond_update_interval(1);
  return engine_sptr;
}

void
TestPreconditionedGradient::check_refused(const std::string& what,
                                          const shared_ptr<GeneralisedGradientEstimator<target_type>>& estimator_sptr,
                                          const shared_ptr<GeneralisedPreconditioner<target_type>>& preconditioner_sptr,
                                          const shared_ptr<GeneralisedPrior<target_type>>& prior_sptr)
{
  std::cerr << "\nYou should now see a message explaining why " << what << " is refused\n";
  shared_ptr<PreconditionedGradientReconstruction<target_type>> engine_sptr(this->construct_engine(prior_sptr, 4));
  engine_sptr->set_gradient_estimator_sptr(estimator_sptr);
  engine_sptr->set_preconditioner_sptr(preconditioner_sptr);
  shared_ptr<target_type> target_sptr(this->_input_density_sptr->get_empty_copy());
  target_sptr->fill(1.F);
  bool refused = false;
  try
    {
      refused = (engine_sptr->set_up(target_sptr) == Succeeded::no);
    }
  catch (...)
    {
      // an exception is an acceptable way of refusing too
      refused = true;
    }
  check(refused, what + " should be refused at set_up()");
}

void
TestPreconditionedGradient::test_contracts()
{
  std::cerr << "\n--- contracts checked at set_up() ---\n";

  shared_ptr<GeneralisedPrior<target_type>> no_prior;

  // no axis selected: the engine has no default, on purpose
  {
    shared_ptr<GeneralisedGradientEstimator<target_type>> no_estimator;
    shared_ptr<GeneralisedPreconditioner<target_type>> no_preconditioner;
    this->check_refused("an engine without any axis", no_estimator, no_preconditioner, no_prior);
  }

  // scale contract: a subset-scaled preconditioner needs the per-subset estimator
  {
    shared_ptr<EMPreconditioner<target_type>> em_sptr(new EMPreconditioner<target_type>);
    em_sptr->set_use_subset_sensitivity(true);
    shared_ptr<GeneralisedGradientEstimator<target_type>> full_sptr(new FullGradientEstimator<target_type>);
    this->check_refused("a full gradient with a subset-scaled preconditioner", full_sptr, em_sptr, no_prior);
  }

  // prior capability: the Hessian diagonal needs a prior that computes one
  {
    shared_ptr<GeneralisedPrior<target_type>> quadratic_sptr(new QuadraticPrior<float>(false, 1.F));
    shared_ptr<GeneralisedPreconditioner<target_type>> hessian_sptr(new HessianDiagonalPreconditioner<target_type>);
    shared_ptr<GeneralisedGradientEstimator<target_type>> os_sptr(new OrderedSubsetsGradientEstimator<target_type>);
    this->check_refused("the Hessian diagonal with a prior that does not provide one", os_sptr, hessian_sptr, quadratic_sptr);
  }

  // prior capability: SPS needs a parabolic surrogate, which no RDP provides
  {
    shared_ptr<GeneralisedPrior<target_type>> rdp_sptr(new GibbsRelativeDifferencePenalty<float>);
    shared_ptr<GeneralisedPreconditioner<target_type>> sps_sptr(new SPSPreconditioner<target_type>);
    shared_ptr<GeneralisedGradientEstimator<target_type>> os_sptr(new OrderedSubsetsGradientEstimator<target_type>);
    this->check_refused("SPS with a prior that provides no parabolic surrogate", os_sptr, sps_sptr, rdp_sptr);
  }
}

//! required by the base class: the MLEM composition (full gradient, EM, unit step, one subset)
void
TestPreconditionedGradient::construct_reconstructor()
{
  this->_recon_sptr = this->construct_engine(shared_ptr<GeneralisedPrior<target_type>>(), /*num_subsets=*/1);
  this->recon().set_gradient_estimator_sptr(
      shared_ptr<GeneralisedGradientEstimator<target_type>>(new FullGradientEstimator<target_type>));
  this->recon().set_preconditioner_sptr(shared_ptr<GeneralisedPreconditioner<target_type>>(new EMPreconditioner<target_type>));
  this->recon().set_initial_step_size(1.);
  this->recon().set_step_size_decay(0.);
  this->recon().set_num_subiterations(20);
}

double
TestPreconditionedGradient::max_relative_difference_in_fov(const target_type& a, const target_type& b) const
{
  // the field of view as the EM preconditioner defines it: A^T 1 above a fraction of its maximum
  const target_type& sensitivity = this->_objective_function_sptr->get_subset_sensitivity(0);
  const float threshold = 1e-3F * sensitivity.find_max();
  const float scale = b.find_max();

  double result = 0.;
  auto a_iter = a.begin_all_const();
  auto b_iter = b.begin_all_const();
  auto sensitivity_iter = sensitivity.begin_all_const();
  for (; a_iter != a.end_all_const(); ++a_iter, ++b_iter, ++sensitivity_iter)
    if (*sensitivity_iter > threshold)
      result = std::max(result, std::fabs(static_cast<double>(*a_iter) - static_cast<double>(*b_iter)) / scale);
  return result;
}

double
TestPreconditionedGradient::mean_relative_difference_in_fov(const target_type& a, const target_type& b) const
{
  const target_type& sensitivity = this->_objective_function_sptr->get_subset_sensitivity(0);
  const float threshold = 1e-3F * sensitivity.find_max();
  const float scale = b.find_max();

  double sum = 0.;
  std::size_t count = 0;
  auto a_iter = a.begin_all_const();
  auto b_iter = b.begin_all_const();
  auto sensitivity_iter = sensitivity.begin_all_const();
  for (; a_iter != a.end_all_const(); ++a_iter, ++b_iter, ++sensitivity_iter)
    if (*sensitivity_iter > threshold)
      {
        sum += std::fabs(static_cast<double>(*a_iter) - static_cast<double>(*b_iter)) / scale;
        ++count;
      }
  return count == 0 ? 0. : sum / count;
}

void
TestPreconditionedGradient::test_against_OSMAPOSL(const std::string& what, int num_subsets, bool use_subset_sensitivity)
{
  std::cerr << "\n--- " << what << " ---\n";
  const int num_subiterations = 8 * num_subsets;

  // the composed arm
  shared_ptr<PreconditionedGradientReconstruction<target_type>> engine_sptr(
      this->construct_engine(shared_ptr<GeneralisedPrior<target_type>>(), num_subsets));
  {
    shared_ptr<EMPreconditioner<target_type>> em_sptr(new EMPreconditioner<target_type>);
    em_sptr->set_use_subset_sensitivity(use_subset_sensitivity);
    engine_sptr->set_preconditioner_sptr(em_sptr);
  }
  if (use_subset_sensitivity)
    engine_sptr->set_gradient_estimator_sptr(
        shared_ptr<GeneralisedGradientEstimator<target_type>>(new OrderedSubsetsGradientEstimator<target_type>));
  else
    engine_sptr->set_gradient_estimator_sptr(
        shared_ptr<GeneralisedGradientEstimator<target_type>>(new FullGradientEstimator<target_type>));
  engine_sptr->set_initial_step_size(1.);
  engine_sptr->set_step_size_decay(0.);
  engine_sptr->set_num_subiterations(num_subiterations);
  this->_recon_sptr = engine_sptr;

  shared_ptr<target_type> composed_sptr(this->_input_density_sptr->get_empty_copy());
  composed_sptr->fill(1.F);
  this->reconstruct(composed_sptr);

  // the dedicated implementation, same data, same subsets, same number of updates
  this->construct_log_likelihood();
  shared_ptr<OSMAPOSLReconstruction<target_type>> osmaposl_sptr(new OSMAPOSLReconstruction<target_type>);
  osmaposl_sptr->set_objective_function_sptr(this->_objective_function_sptr);
  osmaposl_sptr->set_num_subsets(num_subsets);
  osmaposl_sptr->set_num_subiterations(num_subiterations);
  this->_recon_sptr = osmaposl_sptr;

  shared_ptr<target_type> reference_sptr(this->_input_density_sptr->get_empty_copy());
  reference_sptr->fill(1.F);
  this->reconstruct(reference_sptr);

  // (a) the composition reproduces the dedicated implementation
  const double difference = this->max_relative_difference_in_fov(*composed_sptr, *reference_sptr);
  std::cerr << "max relative difference with OSMAPOSL inside the FOV: " << difference << '\n';
  check_if_less(difference, 1e-3, what + ": composition vs OSMAPOSL");

  // (b) and both actually reconstructed the phantom, so that (a) cannot be passed by two arms
  //     that do nothing. This is a sanity gate, not a statement about accuracy.
  const double error_vs_phantom = this->mean_relative_difference_in_fov(*composed_sptr, *this->_input_density_sptr);
  std::cerr << "mean relative difference with the phantom inside the FOV: " << error_vs_phantom << '\n';
  check_if_less(error_vs_phantom, 0.1, what + ": composition vs phantom");
}

void
TestPreconditionedGradient::run_tests()
{
  std::cerr << "Tests for PreconditionedGradientReconstruction\n";

  try
    {
      this->construct_input_data();
      this->test_contracts();
      this->test_against_OSMAPOSL("Full Gradient + EM + unit step: this is MLEM", 1, /*use_subset_sensitivity=*/false);
      this->test_against_OSMAPOSL("Ordered Subsets + subset EM + unit step: this is OSEM", 4, /*use_subset_sensitivity=*/true);
    }
  catch (const std::exception& error)
    {
      std::cerr << "\nHere's the error:\n\t" << error.what() << "\n\n";
      everything_ok = false;
    }
  catch (...)
    {
      everything_ok = false;
    }
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int
main(int argc, char** argv)
{
  if (argc < 1 || argc > 4)
    {
      std::cerr << "\nUsage: " << argv[0] << " [projector_pair_filename [template_proj_data [image]]]\n"
                << "projector_pair_filename (optional) can be used to specify the projectors\n"
                << "  if set to an empty string, the default ray-tracing matrix will be used.\n"
                << "template_proj_data (optional) will serve as a template, but is otherwise not used.\n"
                << "image (optional) has to be compatible with projection data and currently at zoom=1\n";
      return EXIT_FAILURE;
    }

  TestPreconditionedGradient test(argc > 1 ? argv[1] : "", argc > 2 ? argv[2] : "", argc > 3 ? argv[3] : "");

  if (test.is_everything_ok())
    test.run_tests();

  return test.main_return_value();
}
