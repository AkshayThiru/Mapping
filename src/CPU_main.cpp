// g++ -std=c++11 CPU_main.cpp -o CPU_main -lrealsense2 -lboost_iostreams -lboost_system -lboost_filesystem `pkg-config opencv --cflags --libs` -lpthread

#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <unistd.h>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <time.h>

#include "../include/Voxel.hpp"
#include "../include/Logging.hpp"




int main(int argc, char const *argv[])
{
	std::atomic_bool alive {true};

    /* Map Front End */
    Map_FE * F = new CPU_FE();

    /* Camera Initialization */
    Camera C;
    Bool_Init bC = C.Init();
    if (bC.t265 && bC.d435)
        std::cout << "Cameras initialized\n";
    else 
        std::cout << "Atleast one camera is not connected\n";

    /* Logger Initialization */
    Logger L;
    L.Init();

    /* Thread for checking exit condition */

    std::thread exit_check([&]() {
        while (alive) {
            if (std::cin.get() == ' ') {
                cv::destroyAllWindows();
                alive = false;
            }
        }
    });

    /* Thread for receiving frames and storing them as video and csv files */ 

    std::thread rxFrame([&]() {
        while (alive) {
            auto sleep_start = std::chrono::high_resolution_clock::now();

			auto tframe = C.pipelines[0].wait_for_frames();
			auto dframe = C.pipelines[1].wait_for_frames();

			auto t = tframe.first_or_default(RS2_STREAM_POSE);
			auto d = dframe.get_depth_frame();

			if (!t || !d)
				continue;

			C.t_queue.enqueue(tframe);
			C.d_queue.enqueue(dframe);

            // sleep for remaining time
            auto time_sleep = std::chrono::high_resolution_clock::now() - sleep_start;
            double time_s = std::chrono::duration_cast<std::chrono::milliseconds>(time_sleep).count();
            if ((1000.0/INPUT_RATE)-time_s > 0){
                usleep((1000.0/INPUT_RATE-time_s) * 1000);
            }
            // std::cout << time_s << "\n";
		}
    });


    rs2::frameset t_frameset, d_frameset;
    auto start = std::chrono::high_resolution_clock::now();

    while (alive) {
    	C.t_queue.poll_for_frame(&t_frameset);
    	C.d_queue.poll_for_frame(&d_frameset);

    	if (t_frameset && d_frameset) {
    		auto depthFrame = d_frameset.get_depth_frame();
    		auto poseFrame  = t_frameset.first_or_default(RS2_STREAM_POSE);

    		cv::Mat depth(cv::Size(w, h), CV_16UC1, (void *)depthFrame.get_data(), cv::Mat::AUTO_STEP);
    		auto pose = poseFrame.as<rs2::pose_frame>().get_pose_data();

            /* update global map */
            F->Update (C, pose, depth);
            /*                   */

    		auto elapsed = std::chrono::high_resolution_clock::now() - start;
            float milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
            //std::cout << milliseconds << "\n";

            L.Log(&C, &pose, &depth);

    	}

    	start = std::chrono::high_resolution_clock::now();

    }

    rxFrame.join();

    L.Close(&C, F);

    std::cout << "Program terminated sucessfully\n";
	return 0;
	
}