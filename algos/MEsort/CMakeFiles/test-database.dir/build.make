# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/vboxuser/src/algos/MEsort

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/vboxuser/src/algos/MEsort

# Include any dependencies generated for this target.
include CMakeFiles/test-database.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/test-database.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/test-database.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test-database.dir/flags.make

CMakeFiles/test-database.dir/test_database.cpp.o: CMakeFiles/test-database.dir/flags.make
CMakeFiles/test-database.dir/test_database.cpp.o: test_database.cpp
CMakeFiles/test-database.dir/test_database.cpp.o: CMakeFiles/test-database.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/vboxuser/src/algos/MEsort/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test-database.dir/test_database.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/test-database.dir/test_database.cpp.o -MF CMakeFiles/test-database.dir/test_database.cpp.o.d -o CMakeFiles/test-database.dir/test_database.cpp.o -c /home/vboxuser/src/algos/MEsort/test_database.cpp

CMakeFiles/test-database.dir/test_database.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test-database.dir/test_database.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vboxuser/src/algos/MEsort/test_database.cpp > CMakeFiles/test-database.dir/test_database.cpp.i

CMakeFiles/test-database.dir/test_database.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test-database.dir/test_database.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vboxuser/src/algos/MEsort/test_database.cpp -o CMakeFiles/test-database.dir/test_database.cpp.s

# Object files for target test-database
test__database_OBJECTS = \
"CMakeFiles/test-database.dir/test_database.cpp.o"

# External object files for target test-database
test__database_EXTERNAL_OBJECTS =

test-database: CMakeFiles/test-database.dir/test_database.cpp.o
test-database: CMakeFiles/test-database.dir/build.make
test-database: CMakeFiles/test-database.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/vboxuser/src/algos/MEsort/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test-database"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test-database.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test-database.dir/build: test-database
.PHONY : CMakeFiles/test-database.dir/build

CMakeFiles/test-database.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test-database.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test-database.dir/clean

CMakeFiles/test-database.dir/depend:
	cd /home/vboxuser/src/algos/MEsort && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vboxuser/src/algos/MEsort /home/vboxuser/src/algos/MEsort /home/vboxuser/src/algos/MEsort /home/vboxuser/src/algos/MEsort /home/vboxuser/src/algos/MEsort/CMakeFiles/test-database.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test-database.dir/depend
