#include "core.h"
#include <fstream>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <algorithm>

#define MILLISECONDS_IN_SECOND  1000.0
#define B_IN_MB  1000000.0
#define BITS_IN_BYTE  8.0
#define RANDOM_SEED  42
#define VIDEO_CHUNCK_LEN  4000.0  // millisec, every time add this amount to buffer
#define MPC_FUTURE_CHUNK_COUNT 8 //define future count
#define DRAIN_BUFFER_SLEEP_TIME  500.0  // millisec
#define PACKET_PAYLOAD_PORTION  0.95
#define LINK_RTT  80  // millisec
#define PACKET_SIZE  1500  // bytes
#define M_IN_K 1000.
#define VIDEO_SIZE_FILE  "./envivio/video_size_"
#define TOTAL_VIDEO_CHUNCK 48
#define BITRATE_LEVELS 6
#define REBUF_PENALTY 4.3
#define SMOOTH_PENALTY 1.0

double VIDEO_BIT_RATE[] = {300,750,1200,1850,2850,4300};

/**
 * Argsort(currently support ascending sort)
 * @tparam T array element type
 * @param array input array
 * @return indices w.r.t sorted array
 */
template<typename T>
std::vector<size_t> argsort(const std::vector<T> &array) {
    std::vector<size_t> indices(array.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&array](int left, int right) -> bool {
                  // sort indices according to corresponding array element
                  return array[left] < array[right];
              });

    return indices;
}

Environment::Environment(std::vector<std::vector<double>> all_cooked_time, 
	std::vector<std::vector<double>> all_cooked_bw, int seed)
{
	this->all_cooked_time = all_cooked_time;
	this->all_cooked_bw = all_cooked_bw;
    srand(seed);

	this->video_chunk_counter = 0;
	this->buffer_size = 0;
	this->buffer_thresh = 60.0 * MILLISECONDS_IN_SECOND;
	// pick a random trace file
	this->trace_idx = rand() % all_cooked_time.size();
	this->cooked_time = this->all_cooked_time[this->trace_idx];
	this->cooked_bw = this->all_cooked_bw[this->trace_idx];

	this->mahimahi_ptr = rand() % (this->cooked_bw.size() - 1) + 1;
	this->last_mahimahi_time = this->cooked_time[this->mahimahi_ptr - 1];

    readChunk(this->video_size);
}

void Environment::readChunk(std::unordered_map<int, std::vector<int>> &chunk_size)
{
	for (auto bitrate = 0; bitrate < BITRATE_LEVELS; bitrate++)
	{
		std::vector<int> tmp;
		chunk_size[bitrate] = tmp;
		std::ifstream fin(VIDEO_SIZE_FILE + std::__cxx11::to_string(bitrate));
		std::string s;
		while (getline(fin, s))
		{
			chunk_size[bitrate].push_back(stoi(s));
		}
		fin.close();
	}
}

// opt = 9
int Environment::get_optimal(int last_bit_rate, int top_k, int horizon)
{
    std::vector<cache_element> rank_buffer;
    auto video_chunk_counter = this->video_chunk_counter;
    rank_buffer.push_back(cache_element(this->mahimahi_ptr, this->last_mahimahi_time, this->buffer_size, last_bit_rate, video_chunk_counter, 0., -1));
    auto video_chunk_counter_end = std::min(video_chunk_counter + horizon, TOTAL_VIDEO_CHUNCK);
    for (auto iter_ = video_chunk_counter; iter_ < video_chunk_counter_end; iter_++)
    {
        std::vector<cache_element> state_arr;
        std::vector<double> rank_arr;
        for (auto &status : rank_buffer)
        {
            auto new_arr = this->get_status(status);
            for (auto &s : new_arr)
            {
                state_arr.push_back(s);
                rank_arr.push_back(-std::get<5>(s));
            }
        }            
        // rank: top-100
        std::vector<cache_element> new_rank_buffer;
        auto same_bitrate = -1;
        bool is_same_bitrate = true;
        auto rank_sort = argsort(rank_arr);
        auto top_len = std::min(top_k, int(rank_sort.size()));
        for (auto rank_idx = 0; rank_idx < top_len; rank_idx++)
        {
            auto rank_ = rank_sort[rank_idx];
            auto stat = state_arr[rank_];
            new_rank_buffer.push_back(stat);
            if (same_bitrate < 0)
            {
                same_bitrate = std::get<6>(stat);
            }
            if (std::get<6>(stat) != same_bitrate)
            {
                is_same_bitrate = false;
            }
        }
        if (is_same_bitrate)
        {
            break;
        }
        rank_buffer = new_rank_buffer;
    }
    return std::get<6>(rank_buffer[0]);
}

std::vector<cache_element> Environment::get_status(cache_element &status)
{
    auto mahimahi_ptr_ = 0.;
    auto last_mahimahi_time_ = 0.;
    auto buffer_size_ = 0.;
    auto last_bit_rate_ = 0;
    auto video_chunk_counter_ = 0;
    auto first_bit_rate_ = 0;
    auto total_reward_ = 0.0;

    std::tie(mahimahi_ptr_, last_mahimahi_time_, buffer_size_, last_bit_rate_, 
        video_chunk_counter_, total_reward_, first_bit_rate_) = status;

    auto tmp_video_chunk_counter = this->video_chunk_counter;
    auto tmp_mahimahi_ptr = this->mahimahi_ptr;
    auto tmp_mahimahi_time = this->last_mahimahi_time;
    auto tmp_buffer_size = this->buffer_size;
    
    std::vector<cache_element> arr;
    for (auto bit_rate = 0; bit_rate < BITRATE_LEVELS; bit_rate++)
    {
        this->mahimahi_ptr = mahimahi_ptr_;
        this->last_mahimahi_time = last_mahimahi_time_;
        this->video_chunk_counter = video_chunk_counter_;
        this->buffer_size = buffer_size_;

        auto delay = 0.;
        auto sleep_time = 0.;
        auto next_buffer_size = 0.; 
        auto rebuf = 0.;
        auto video_chunk_size = 0;
        std::vector<int> next_video_chunk_sizes;
        bool end_of_video;
        auto video_chunk_remain = 0;
		double amenda = 0.9;

        std::tie(delay, sleep_time, next_buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain) = this->get_video_chunk_amenda(bit_rate, amenda, false);
        
        auto reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
            - REBUF_PENALTY * rebuf \
            - SMOOTH_PENALTY * abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate_]) / M_IN_K;
        reward += total_reward_;

        auto current_first_bit_rate_ = 0;
        if (first_bit_rate_ < 0)
            current_first_bit_rate_ = bit_rate;
        else
            current_first_bit_rate_ = first_bit_rate_;

        arr.push_back(cache_element(this->mahimahi_ptr, this->last_mahimahi_time, next_buffer_size * MILLISECONDS_IN_SECOND, \
            bit_rate, this->video_chunk_counter, reward, current_first_bit_rate_));
    }
    
    this->video_chunk_counter = tmp_video_chunk_counter;
    this->mahimahi_ptr = tmp_mahimahi_ptr;
    this->last_mahimahi_time = tmp_mahimahi_time;
    this->buffer_size = tmp_buffer_size;
    return arr;
}

std::tuple<double, double, double, double, double, std::vector<int>, bool, int> 
	Environment::get_video_chunk_amenda(int quality, double amenda, bool switch_trace)
{
    auto bitrate_levels = this->video_size.size();
	auto video_chunk_size = this->video_size[quality][this->video_chunk_counter];
    this->buffer_thresh = 60.0 * MILLISECONDS_IN_SECOND;
	
	// use the delivery opportunity in mahimahi
	auto delay = 0.0;  // in ms
	auto video_chunk_counter_sent = 0;  // in bytes

	while (true)  // download video chunk over mahimahi
	{
		auto throughput = this->cooked_bw[this->mahimahi_ptr] * B_IN_MB / BITS_IN_BYTE * amenda;
		auto duration = this->cooked_time[this->mahimahi_ptr] - this->last_mahimahi_time;

		auto packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION;

		if (video_chunk_counter_sent + packet_payload > video_chunk_size)
		{
			auto fractional_time = (video_chunk_size - video_chunk_counter_sent) / throughput / PACKET_PAYLOAD_PORTION;
			delay += fractional_time;
			this->last_mahimahi_time += fractional_time;
			break;
		}
		video_chunk_counter_sent += packet_payload;
		delay += duration;
		this->last_mahimahi_time = this->cooked_time[this->mahimahi_ptr];
		this->mahimahi_ptr += 1;

		if (this->mahimahi_ptr >= this->cooked_bw.size())
		{
			// loop back in the beginning
			// note: trace file starts with time 0
			this->mahimahi_ptr = 1;
			this->last_mahimahi_time = 0;
		}
	}
	delay *= MILLISECONDS_IN_SECOND;
	delay += LINK_RTT;

	// rebuffer time
	auto rebuf = std::max(delay - this->buffer_size, 0.0);

	// update the buffer
	this->buffer_size = std::max(this->buffer_size - delay, 0.0);

	// add in the new chunk
	this->buffer_size += VIDEO_CHUNCK_LEN;

	// sleep if buffer gets too large
	auto sleep_time = 0.0;
    auto actual_sleep_time = 0.;
	if (this->buffer_size > this->buffer_thresh)
	{
		// exceed the buffer limit
		// we need to skip some network bandwidth here
		// but do not add up the delay
		auto drain_buffer_time = this->buffer_size - this->buffer_thresh;
		sleep_time = std::ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * DRAIN_BUFFER_SLEEP_TIME;
        actual_sleep_time = sleep_time;
		this->buffer_size -= sleep_time;

		while (true)
		{
			auto duration = this->cooked_time[this->mahimahi_ptr] - this->last_mahimahi_time;
			if (duration > sleep_time / MILLISECONDS_IN_SECOND)
			{
				this->last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND;
				break;
			}
			sleep_time -= duration * MILLISECONDS_IN_SECOND;
			this->last_mahimahi_time = this->cooked_time[this->mahimahi_ptr];
			this->mahimahi_ptr += 1;

			if (this->mahimahi_ptr >= this->cooked_bw.size())
			{
				// loop back in the beginning
				// note: trace file starts with time 0
				this->mahimahi_ptr = 1;
				this->last_mahimahi_time = 0;
			}
		}
	}
	// the "last buffer size" return to the controller
	// Note: in old version of dash the lowest buffer is 0.
	// In the new version the buffer always have at least
	// one chunk of video
	auto return_buffer_size = this->buffer_size;

	this->video_chunk_counter += 1;
	auto video_chunk_remain = TOTAL_VIDEO_CHUNCK - this->video_chunk_counter;

	auto end_of_video = false;
	if (this->video_chunk_counter >= TOTAL_VIDEO_CHUNCK)
	{
		end_of_video = true;
		this->buffer_size = 0;
		this->video_chunk_counter = 0;

        // pick a random trace file
        if (switch_trace)
		    this->trace_idx = rand() % all_cooked_time.size();
		this->cooked_time = this->all_cooked_time[this->trace_idx];
		this->cooked_bw = this->all_cooked_bw[this->trace_idx];

		// randomize the start point of the video
		// note: trace file starts with time 0
		this->mahimahi_ptr = rand() % (this->cooked_bw.size() - 1) + 1;
		this->last_mahimahi_time = this->cooked_time[this->mahimahi_ptr - 1];

		this->buffer_thresh = 60.0 * MILLISECONDS_IN_SECOND;
	}
	std::vector<int> next_video_chunk_sizes;
	for (auto i = 0; i< bitrate_levels; i++)
	{
		next_video_chunk_sizes.push_back(this->video_size[i][this->video_chunk_counter]);
	}
	return std::tuple<double, double, double, double, double, std::vector<int>, bool, int>
		(delay, actual_sleep_time, return_buffer_size / MILLISECONDS_IN_SECOND, rebuf / MILLISECONDS_IN_SECOND,
		video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain);
}

std::tuple<double, double, double, double, double, std::vector<int>, bool, int> 
	Environment::get_video_chunk(int quality, bool switch_trace)
{
    auto bitrate_levels = this->video_size.size();
	auto video_chunk_size = this->video_size[quality][this->video_chunk_counter];
    this->buffer_thresh = 60.0 * MILLISECONDS_IN_SECOND;
	
	// use the delivery opportunity in mahimahi
	auto delay = 0.0;  // in ms
	auto video_chunk_counter_sent = 0;  // in bytes

	while (true)  // download video chunk over mahimahi
	{
		auto throughput = this->cooked_bw[this->mahimahi_ptr] * B_IN_MB / BITS_IN_BYTE;
		auto duration = this->cooked_time[this->mahimahi_ptr] - this->last_mahimahi_time;

		auto packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION;

		if (video_chunk_counter_sent + packet_payload > video_chunk_size)
		{
			auto fractional_time = (video_chunk_size - video_chunk_counter_sent) / throughput / PACKET_PAYLOAD_PORTION;
			delay += fractional_time;
			this->last_mahimahi_time += fractional_time;
			break;
		}
		video_chunk_counter_sent += packet_payload;
		delay += duration;
		this->last_mahimahi_time = this->cooked_time[this->mahimahi_ptr];
		this->mahimahi_ptr += 1;

		if (this->mahimahi_ptr >= this->cooked_bw.size())
		{
			// loop back in the beginning
			// note: trace file starts with time 0
			this->mahimahi_ptr = 1;
			this->last_mahimahi_time = 0;
		}
	}
	delay *= MILLISECONDS_IN_SECOND;
	delay += LINK_RTT;

	// rebuffer time
	auto rebuf = std::max(delay - this->buffer_size, 0.0);

	// update the buffer
	this->buffer_size = std::max(this->buffer_size - delay, 0.0);

	// add in the new chunk
	this->buffer_size += VIDEO_CHUNCK_LEN;

	// sleep if buffer gets too large
	auto sleep_time = 0.0;
    auto actual_sleep_time = 0.;
	if (this->buffer_size > this->buffer_thresh)
	{
		// exceed the buffer limit
		// we need to skip some network bandwidth here
		// but do not add up the delay
		auto drain_buffer_time = this->buffer_size - this->buffer_thresh;
		sleep_time = std::ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * DRAIN_BUFFER_SLEEP_TIME;
        actual_sleep_time = sleep_time;
		this->buffer_size -= sleep_time;

		while (true)
		{
			auto duration = this->cooked_time[this->mahimahi_ptr] - this->last_mahimahi_time;
			if (duration > sleep_time / MILLISECONDS_IN_SECOND)
			{
				this->last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND;
				break;
			}
			sleep_time -= duration * MILLISECONDS_IN_SECOND;
			this->last_mahimahi_time = this->cooked_time[this->mahimahi_ptr];
			this->mahimahi_ptr += 1;

			if (this->mahimahi_ptr >= this->cooked_bw.size())
			{
				// loop back in the beginning
				// note: trace file starts with time 0
				this->mahimahi_ptr = 1;
				this->last_mahimahi_time = 0;
			}
		}
	}
	// the "last buffer size" return to the controller
	// Note: in old version of dash the lowest buffer is 0.
	// In the new version the buffer always have at least
	// one chunk of video
	auto return_buffer_size = this->buffer_size;

	this->video_chunk_counter += 1;
	auto video_chunk_remain = TOTAL_VIDEO_CHUNCK - this->video_chunk_counter;

	auto end_of_video = false;
	if (this->video_chunk_counter >= TOTAL_VIDEO_CHUNCK)
	{
		end_of_video = true;
		this->buffer_size = 0;
		this->video_chunk_counter = 0;

        // pick a random trace file
        if (switch_trace)
		    this->trace_idx = rand() % all_cooked_time.size();
		this->cooked_time = this->all_cooked_time[this->trace_idx];
		this->cooked_bw = this->all_cooked_bw[this->trace_idx];

		// randomize the start point of the video
		// note: trace file starts with time 0
		this->mahimahi_ptr = rand() % (this->cooked_bw.size() - 1) + 1;
		this->last_mahimahi_time = this->cooked_time[this->mahimahi_ptr - 1];

		this->buffer_thresh = 60.0 * MILLISECONDS_IN_SECOND;
	}
	std::vector<int> next_video_chunk_sizes;
	for (auto i = 0; i< bitrate_levels; i++)
	{
		next_video_chunk_sizes.push_back(this->video_size[i][this->video_chunk_counter]);
	}
	return std::tuple<double, double, double, double, double, std::vector<int>, bool, int>
		(delay, actual_sleep_time, return_buffer_size / MILLISECONDS_IN_SECOND, rebuf / MILLISECONDS_IN_SECOND,
		video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain);
}

PYBIND11_MODULE(libcore, m) {
    pybind11::class_<Environment>(m, "Environment")
        .def(pybind11::init<std::vector<std::vector<double>>, std::vector<std::vector<double>>, int>())
        .def_readwrite("trace_idx", &Environment::trace_idx)
        .def("get_optimal", &Environment::get_optimal)
        .def("get_video_chunk", &Environment::get_video_chunk);
}