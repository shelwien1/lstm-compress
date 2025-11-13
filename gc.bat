@echo off

set path=C:\VC\link64
rem C:\VC2019\bin\amd64;
set LIB=C:\VC2019\lib\amd64;C:\VC2019\sdk\Lib\x64;C:\VC2019\ucrt\x64
set INCLUDE=

set incs=-std=c++17 -DSTRICT -DNDEBUG -DWIN32 -I../Lib ^
-D_CRT_SECURE_NO_WARNINGS ^
-D_CRT_SECURE_NO_DEPRECATE ^
-D_CRT_DISABLE_PERFCRIT_LOCKS ^
-D_CRT_NONSTDC_NO_DEPRECATE ^
-D_SECURE_SCL=0 ^
-D_ITERATOR_DEBUG_LEVEL=0 ^
-D_SECURE_SCL_THROWS=0 ^
-D_HAS_ITERATOR_DEBUGGING=0 ^
-DCOMMON_SKIP_BSF ^
-IC:\VC2019\include-no-intrin -IC:\VC2019\sdk\include -IC:\VC2019\ucrt\include ^
-DWIN32_LEAN_AND_MEAN ^
-DVC_EXTRALEAN -I./mim-include -DMI_BUILD_RELEASE -DMI_CMAKE_BUILD_TYPE=release -DMI_STATIC_LIB 
rem -Dalign_val_t=size_t
rem -IC:\clangJ104x\lib\clang\19\include ^
rem -IC:\clangK10x\lib\clang\20\include\ ^

rem -DUNICODE -D_UNICODE 

set opts=-fomit-frame-pointer -fno-stack-protector -fno-stack-check -fgnu-keywords -fstrict-aliasing -ffast-math ^
-Wmsvc-not-found -Wno-nonportable-include-path -Wno-pragma-pack -Wno-ignored-pragma-intrinsic ^
-Wno-ignored-attributes -Wno-expansion-to-defined -fno-stack-clash-protection -flto -fuse-ld=lld

rem -fwhole-program -fno-unsafe-math-optimizations 

set targ=skylake
set arch=-march=%targ% -mtune=%targ% -mavx512f -fms-compatibility -fms-compatibility-version=19 -fms-extensions -Wno-ignored-attributes -m64 
rem -mno-avx2

set gcc=C:\clang900\bin\clang++.exe
set gcc=C:\clang801x\bin\clang++.exe
set gcc=C:\clangB00x\bin\clang++.exe
set gcc=C:\clangJ104x\bin\clang++.exe -include "C:\VC2019\include\vcruntime_new.h" -IC:\clangJ104x\lib\clang\19\include
set gcc=C:\clangK10x\bin\clang++.exe -include "C:\VC2019\include\vcruntime_new.h" -IC:\clangK10x\lib\clang\20\include
set gcc=C:\clangL15x\bin\clang++.exe -include "C:\VC2019\include\vcruntime_new.h" -IC:\clangL15x\lib\clang\21\include 

rem -include stdarg.h -include "C:\VC2019\include\vadefs1.h" -include "C:\VC2019\include\stdint.h" -fprofile-use
rem -include vadefs1.h -Dva_list=char* -Duintptr_t=intptr_t -D_M_X64 
rem -Dva_list=void* -D__crt_va_start(x,y)= -D__crt_va_end(x)= -D__builtin_va_start(x,y)= -D__builtin_va_end(x)=

del *.exe *.o

rem %gcc% -v -O9 %arch% %incs% %opts% test.cpp -o test.exe

copy /b C:\VC2019\bin\amd64\link.exe .

for /D %%a in (.) do set DIRNAM=%%~na

rem %gcc% -std=c++11 -O9 -s %incs% %opts% zdelta.cpp ../zstd145/zstd.o -o zdelta.exe
%gcc% -s -std=c++17 -O9 -Ofast %arch% %incs% %opts% -static "-D__DIRNAM__=%DIRNAM%" coder.cpp mim-src/static.c -ladvapi32 -o coder.exe
rem -o coder.exe
rem -S -fverbose-asm -mllvm --x86-asm-syntax=intel 

del link.exe

del *.o



