cloc . --exclude-dir=build

mkdir -p build
cd build
cmake ..
make USE_THREAD=8
