# Warnings for Katal's own targets. Applied per target and PRIVATE, so they
# never reach a consumer through the exported Katal::katal.
option(KATAL_WARNINGS_AS_ERRORS
       "Treat compiler warnings in Katal's own targets as errors" OFF)

function(katal_target_warnings target)
  if(MSVC)
    target_compile_options(
      ${target} PRIVATE /W4 $<$<BOOL:${KATAL_WARNINGS_AS_ERRORS}>:/WX>)
  else()
    target_compile_options(
      ${target} PRIVATE -Wall -Wextra -Wpedantic
                        $<$<BOOL:${KATAL_WARNINGS_AS_ERRORS}>:-Werror>)
  endif()
endfunction()
