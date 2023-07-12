c++ -O4 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) core.cc -o libcore$(python3-config --extension-suffix)
cp *.so ../
echo 'Done'