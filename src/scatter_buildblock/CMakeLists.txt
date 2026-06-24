
#
#

set(dir scatter_buildblock)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)
option(USE_SCATTER_OPTI "Use optimised scatter integrals (single_scatter_integrals_opti.cxx)" OFF)

if(USE_SCATTER_OPTI)
  set(_scatter_integrals_src single_scatter_integrals_opti.cxx)
  message(STATUS "scatter_buildblock: using OPTIMISED single_scatter_integrals_opti.cxx")
else()
  set(_scatter_integrals_src single_scatter_integrals.cxx)
  message(STATUS "scatter_buildblock: using standard single_scatter_integrals.cxx")
endif()

set(${dir_LIB_SOURCES}
	sample_scatter_points.cxx
	single_scatter_estimate.cxx
	${_scatter_integrals_src}
	scatter_detection_modelling.cxx
        cached_single_scatter_integrals.cxx
	scatter_estimate_for_one_scatter_point.cxx
	upsample_and_fit_scatter_estimate.cxx
	ScatterEstimation.cxx
	CreateTailMaskFromACFs.cxx
	ScatterSimulation.cxx
	SingleScatterSimulation.cxx
)
#$(dir)_REGISTRY_SOURCES:= scatter_buildblock_registries


include(stir_lib_target)
target_link_libraries(scatter_buildblock PUBLIC ${STIR_RECON_BUILDBLOCK_LIB})
