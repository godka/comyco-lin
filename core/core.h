#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using cache_element = std::tuple<int, double, double, int, int, double, int>;
class Environment
{
public:
	//void start();
    Environment(std::vector<std::vector<double>> all_cooked_time, 
	    std::vector<std::vector<double>> all_cooked_bw, int seed = 42);
    ~Environment(){;};
    int trace_idx;
	std::tuple<double, double, double, double, double, std::vector<int>, bool, int> 
        get_video_chunk(int quality, bool switch_trace);
    std::unordered_map<int, std::vector<int>> video_size;
    int get_optimal(int last_bit_rate, int top_k, int horizon);
protected:
    std::vector<std::vector<double>> all_cooked_bw;
    std::vector<std::vector<double>> all_cooked_time;
    int video_chunk_counter;
    double buffer_thresh;
    double buffer_size;
    std::vector<double> cooked_time;
    std::vector<double> cooked_bw;
    int mahimahi_start_ptr;
    int mahimahi_ptr;
    double last_mahimahi_time;
private:
    std::vector<cache_element> get_status(cache_element &status);
    void readChunk(std::unordered_map<int, std::vector<int>> &chunk_size);
	std::tuple<double, double, double, double, double, std::vector<int>, bool, int> 
        get_video_chunk_amenda(int quality, double amenda, bool switch_trace);
};