CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
FIND_PACKAGE(Torch REQUIRED)

SET(src)
SET(luasrc pprint/init.lua)

ADD_TORCH_PACKAGE(pprint "${src}" "${luasrc}" "A pretty printer for Lua and Torch.")
