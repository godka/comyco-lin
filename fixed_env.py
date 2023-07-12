import numpy as np

MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 42
VIDEO_CHUNCK_LEN = 4000.0  # millisec, every time add this amount to buffer
BITRATE_LEVELS = 6
TOTAL_VIDEO_CHUNCK = 48
BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
M_IN_K = 1000.
REBUF_PENALTY = 4.3
SMOOTH_PENALTY = 1.0
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec
PACKET_SIZE = 1500  # bytes
VIDEO_SIZE_FILE = './envivio/video_size_'


class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, random_seed=RANDOM_SEED):
        assert len(all_cooked_time) == len(all_cooked_bw)

        np.random.seed(random_seed)

        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw

        self.video_chunk_counter = 0
        self.buffer_size = 0

        # pick a random trace file
        self.trace_idx = 0
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        self.mahimahi_start_ptr = 1
        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = 1
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.video_size = {}  # in bytes
        for bitrate in range(BITRATE_LEVELS):
            self.video_size[bitrate] = []
            with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))

    def get_optimal(self, last_bit_rate, top_k = 100):
        rank_buffer = []
        video_chunk_counter = self.video_chunk_counter
        rank_buffer.append([self.mahimahi_ptr, self.last_mahimahi_time, self.buffer_size, last_bit_rate, video_chunk_counter, 0., -1])
        for iter_ in range(video_chunk_counter, TOTAL_VIDEO_CHUNCK):
            state_arr, rank_arr = [], []
            for status in rank_buffer:
                new_arr = self.get_status(status)
                for s in new_arr:
                    state_arr.append(s)
                    rank_arr.append(s[-2])                
            # rank: top-100
            rank_buffer = []
            same_bitrate = None
            is_same_bitrate = True
            rank_arr = np.array(rank_arr)
            rank_sort = np.argsort(-rank_arr)
            for rank_ in rank_sort[:top_k]:
                stat = state_arr[rank_]
                rank_buffer.append(stat)
                if same_bitrate is None:
                    same_bitrate = stat[-1]
                if stat[-1] != same_bitrate:
                    is_same_bitrate = False
            if is_same_bitrate:
                break
        return rank_buffer[0][-1]

    def get_status(self, status):
        mahimahi_ptr, last_mahimahi_time, buffer_size, last_bit_rate, video_chunk_counter, total_reward, first_bit_rate = status
        tmp_video_chunk_counter = self.video_chunk_counter
        tmp_mahimahi_ptr = self.mahimahi_ptr
        tmp_mahimahi_time = self.last_mahimahi_time
        tmp_buffer_size = self.buffer_size
        
        arr = []
        for bit_rate in range(BITRATE_LEVELS):
            # print(video_chunk_counter)
            self.mahimahi_ptr = mahimahi_ptr
            self.last_mahimahi_time = last_mahimahi_time
            self.video_chunk_counter = video_chunk_counter
            self.buffer_size = buffer_size
            # print(bit_rate, self.mahimahi_ptr, self.last_mahimahi_time, len(self.cooked_bw))
            delay, sleep_time, next_buffer_size, rebuf, \
                video_chunk_size, next_video_chunk_sizes, \
                end_of_video, video_chunk_remain = self.get_video_chunk(bit_rate, switch_trace=False)
            
            # reward is video quality - rebuffer penalty - smooth penalty
            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                - REBUF_PENALTY * rebuf \
                - SMOOTH_PENALTY * abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
            # print(rebuf, reward)
            reward += total_reward

            if first_bit_rate < 0:
                first_bit_rate_ = bit_rate
            else:
                first_bit_rate_ = first_bit_rate

            arr.append([self.mahimahi_ptr, self.last_mahimahi_time, next_buffer_size * MILLISECONDS_IN_SECOND, \
                bit_rate, self.video_chunk_counter, reward, first_bit_rate_])

        self.video_chunk_counter = tmp_video_chunk_counter
        self.mahimahi_ptr = tmp_mahimahi_ptr
        self.last_mahimahi_time = tmp_mahimahi_time
        self.buffer_size = tmp_buffer_size
        return arr

    def get_video_chunk(self, quality, switch_trace=True):

        assert quality >= 0
        assert quality < BITRATE_LEVELS

        video_chunk_size = self.video_size[quality][self.video_chunk_counter]

        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms
        video_chunk_counter_sent = 0  # in bytes

        while True:  # download video chunk over mahimahi
            throughput = self.cooked_bw[self.mahimahi_ptr] \
                         * B_IN_MB / BITS_IN_BYTE
            duration = self.cooked_time[self.mahimahi_ptr] \
                       - self.last_mahimahi_time

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_chunk_counter_sent + packet_payload > video_chunk_size:

                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                  throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                break

            video_chunk_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0

        delay *= MILLISECONDS_IN_SECOND
        delay += LINK_RTT

        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

        # add in the new chunk
        self.buffer_size += VIDEO_CHUNCK_LEN

        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size > BUFFER_THRESH:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - BUFFER_THRESH
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
                         DRAIN_BUFFER_SLEEP_TIME
            self.buffer_size -= sleep_time

            while True:
                duration = self.cooked_time[self.mahimahi_ptr] \
                           - self.last_mahimahi_time
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = 0

        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.buffer_size

        self.video_chunk_counter += 1
        video_chunk_remain = TOTAL_VIDEO_CHUNCK - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= TOTAL_VIDEO_CHUNCK:
            end_of_video = True
            self.buffer_size = 0
            self.video_chunk_counter = 0
            
            if switch_trace:
                self.trace_idx += 1
                if self.trace_idx >= len(self.all_cooked_time):
                    self.trace_idx = 0            

            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]

            # randomize the start point of the video
            # note: trace file starts with time 0
            self.mahimahi_ptr = self.mahimahi_start_ptr
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        next_video_chunk_sizes = []
        for i in range(BITRATE_LEVELS):
            next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])

        return delay, \
            sleep_time, \
            return_buffer_size / MILLISECONDS_IN_SECOND, \
            rebuf / MILLISECONDS_IN_SECOND, \
            video_chunk_size, \
            next_video_chunk_sizes, \
            end_of_video, \
            video_chunk_remain
