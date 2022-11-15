cloc . --exclude-dir=build

mkdir -p build
cd build
cmake ..
make clean
make
