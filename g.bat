@echo off

del coder.exe

set incs=-DNDEBUG -DSTRICT -DWIN32 -I./mim-include -DMI_BUILD_RELEASE -DMI_CMAKE_BUILD_TYPE=release -DMI_STATIC_LIB

set opts=-fomit-frame-pointer -fno-stack-protector -fno-stack-check -fno-check-new ^
-fno-rtti -fno-exceptions -fpermissive -fstrict-aliasing -ftree-vectorize 
rem -foptimize-crc -DOPTCRC

set gcc=C:\MinGWB10x\bin\g++.exe -m64 -march=skylake
set gcc=C:\MinGWE20x\bin\gcc.exe -m64 -march=k8 -mtune=k8
set gcc=C:\MinGWE20x\bin\gcc.exe -m64 -march=native -mtune=native
set gcc=C:\MinGWF20x\bin\g++.exe -m64 -march=k8 -mtune=k8
set gcc=C:\MinGWF20x\bin\g++.exe -m64 -march=skylake -mtune=skylake
set path=%gcc%\..\

del *.exe *.o

%gcc% -s -std=gnu++17 -O3 -Ofast %incs% %opts% -static coder.cpp mim-src/static.c -o coder.exe

rem %gcc% -g -std=gnu++17 -O0 -fno-inline %incs% %opts% -static coder.cpp -o coder.exe -fno-fast-math 
