# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.11

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/bilboi/programming/crf

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/bilboi/programming/crf/build

# Include any dependencies generated for this target.
include src/CMakeFiles/crfmu.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/crfmu.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/crfmu.dir/flags.make

src/CMakeFiles/crfmu.dir/crfmu.cpp.o: src/CMakeFiles/crfmu.dir/flags.make
src/CMakeFiles/crfmu.dir/crfmu.cpp.o: ../src/crfmu.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bilboi/programming/crf/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/crfmu.dir/crfmu.cpp.o"
	cd /home/bilboi/programming/crf/build/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/crfmu.dir/crfmu.cpp.o -c /home/bilboi/programming/crf/src/crfmu.cpp

src/CMakeFiles/crfmu.dir/crfmu.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/crfmu.dir/crfmu.cpp.i"
	cd /home/bilboi/programming/crf/build/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bilboi/programming/crf/src/crfmu.cpp > CMakeFiles/crfmu.dir/crfmu.cpp.i

src/CMakeFiles/crfmu.dir/crfmu.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/crfmu.dir/crfmu.cpp.s"
	cd /home/bilboi/programming/crf/build/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bilboi/programming/crf/src/crfmu.cpp -o CMakeFiles/crfmu.dir/crfmu.cpp.s

# Object files for target crfmu
crfmu_OBJECTS = \
"CMakeFiles/crfmu.dir/crfmu.cpp.o"

# External object files for target crfmu
crfmu_EXTERNAL_OBJECTS =

src/crfmu: src/CMakeFiles/crfmu.dir/crfmu.cpp.o
src/crfmu: src/CMakeFiles/crfmu.dir/build.make
src/crfmu: src/CMakeFiles/crfmu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/bilboi/programming/crf/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable crfmu"
	cd /home/bilboi/programming/crf/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/crfmu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/crfmu.dir/build: src/crfmu

.PHONY : src/CMakeFiles/crfmu.dir/build

src/CMakeFiles/crfmu.dir/clean:
	cd /home/bilboi/programming/crf/build/src && $(CMAKE_COMMAND) -P CMakeFiles/crfmu.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/crfmu.dir/clean

src/CMakeFiles/crfmu.dir/depend:
	cd /home/bilboi/programming/crf/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/bilboi/programming/crf /home/bilboi/programming/crf/src /home/bilboi/programming/crf/build /home/bilboi/programming/crf/build/src /home/bilboi/programming/crf/build/src/CMakeFiles/crfmu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/crfmu.dir/depend

