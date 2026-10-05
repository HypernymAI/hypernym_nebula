# Embed a binary file as a C byte array: cmake -DIN=<file> -DOUT=<header> -P embed.cmake
file(READ "${IN}" HEX HEX)
string(LENGTH "${HEX}" N)
math(EXPR NBYTES "${N} / 2")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," BYTES "${HEX}")
file(WRITE "${OUT}" "// generated from shield.metallib by embed.cmake — do not edit\n#pragma once\nstatic const unsigned char kShieldMetallib[] = {${BYTES}};\nstatic const unsigned long kShieldMetallibSize = ${NBYTES};\n")
