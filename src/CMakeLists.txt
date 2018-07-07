#add_executable(crf_run main.cpp)
add_executable(crf_r graphicalmodel.cpp)


add_library(crf SHARED crfmu.cpp graphicalmodel.cpp)
set_target_properties(crf PROPERTIES PREFIX "")

add_executable(infer_test infer.cpp)
