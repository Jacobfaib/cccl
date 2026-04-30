string(APPEND TEST_PREFIX ".transform")

cudax_add_catch2_test(test_target "${TEST_PREFIX}.transform" transform.cu)
