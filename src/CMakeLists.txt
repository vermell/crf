#add_executable(crf_run main.cpp)
add_executable(crf_r graphicalmodel.cpp)


ADD_LIBRARY(
    qpbo
    ${QPBO_DIR}/QPBO.cpp
    ${QPBO_DIR}/QPBO_extra.cpp
    ${QPBO_DIR}/QPBO_maxflow.cpp
	${QPBO_DIR}/QPBO_postprocessing.cpp
)

add_library(crf SHARED crfmu.cpp graphicalmodel.cpp)
set_target_properties(crf PROPERTIES PREFIX "")

add_executable(infer_test infer.cpp)
TARGET_LINK_LIBRARIES(infer_test qpbo)
