project(CFE_MY_APP C)

# Create the app module
add_cfe_app(my_app fsw/src/my_app.c)

# Include the public API from my_lib to demonstrate how
# to call library-provided functions
add_cfe_app_dependency(my_app sample_lib)#DONT CHANGE LIB

# Add table
add_cfe_tables(my_app fsw/tables/my_app_tbl.c)

target_include_directories(my_app PUBLIC
  fsw/mission_inc
  fsw/platform_inc
)

# If UT is enabled, then add the tests from the subdirectory
# Note that this is an app, and therefore does not provide
# stub functions, as other entities would not typically make 
# direct function calls into this application.
if (ENABLE_UNIT_TESTS)
  add_subdirectory(unit-test)
endif (ENABLE_UNIT_TESTS)
