# Examples

This folder contains standalone example programs for Tiny OpenCV. Each example is a single `.cpp` file.

## Quick Build (Unix-like)

1) Build the library once:

```bash
cmake -S .. -B ../build -DCMAKE_BUILD_TYPE=Release
cmake --build ../build -j
```

2) Compile an example (replace the filename as needed):

```bash
g++ -std=c++11 -I../include basic.cpp ../build/libtiny_opencv.a -o basic
```

Run it:

```bash
./basic
```

Build and run all examples:

```bash
mkdir -p ../build/examples_bin
for f in *.cpp; do
  name=$(basename "$f" .cpp)
  g++ -std=c++11 -I../include "$f" ../build/libtiny_opencv.a -o "../build/examples_bin/$name"
  "../build/examples_bin/$name"
done
```

## CMake Snippet

If you want to build an example with CMake, you can add something like this to a local CMakeLists.txt:

```cmake
add_executable(example_basic basic.cpp)
target_include_directories(example_basic PRIVATE ../include)
target_link_libraries(example_basic PRIVATE ../build/libtiny_opencv.a)
```

## Notes

- Examples are kept minimal and typically omit I/O. Fill input data as needed.
- OpenCV is not required for these examples unless you add your own OpenCV usage.
