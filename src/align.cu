#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>

using namespace std;
using namespace cv;

/*								*/
/*			 Device 			*/
/*								*/

/* Given a point in 3D space, compute the corresponding pixel coordinates in an image with no distortion or forward distortion coefficients produced by the same camera */
/* Device function equivalent for the RealSense function rs2_project_point_to_pixel */
__device__ void project_point_to_pixel(float pixel[2], float intrin[9], int in_model, float point[3]) // intrin: fx(0), fy(1), ppx(2), ppy(3), coeff(4-8)
{
    //assert(intrin->model != 2); // Cannot project to an inverse-distorted image

    float x = point[0] / point[2], y = point[1] / point[2];

    if(in_model == 1)
    {

        float r2  = x*x + y*y;
        float f = 1 + intrin[4]*r2 + intrin[5]*r2*r2 + intrin[8]*r2*r2*r2;
        x *= f;
        y *= f;
        float dx = x + 2*intrin[6]*x*y + intrin[7]*(r2 + 2*x*x);
        float dy = y + 2*intrin[7]*x*y + intrin[6]*(r2 + 2*y*y);
        x = dx;
        y = dy;
    }
    if (in_model == 3)
    {
        float r = sqrtf(x*x + y*y);
        float rd = (float)(1.0f / intrin[4] * atan(2 * r* tan(intrin[4] / 2.0f)));
        x *= rd / r;
        y *= rd / r;
    }

    pixel[0] = x * intrin[0] + intrin[2];
    pixel[1] = y * intrin[1] + intrin[3];
}

/* Given pixel coordinates and depth in an image with no distortion or inverse distortion coefficients, compute the corresponding point in 3D space relative to the same camera */
/* Device function equivalent for the RealSense function rs2_deproject_pixel_to_point */
__device__ void deproject_pixel_to_point(float point[3], float intrin[9], int in_model, float pixel[2], float depth) // intrin: fx(0), fy(1), ppx(2), ppy(3), coeff(4-8)
{
    assert(in_model != 1); // Cannot deproject from a forward-distorted image
    assert(in_model != 3); // Cannot deproject to an ftheta image
    //assert(in_model != RS2_DISTORTION_BROWN_CONRADY); // Cannot deproject to an brown conrady model

    float x = (pixel[0] - intrin[2]) / intrin[0];
    float y = (pixel[1] - intrin[3]) / intrin[1];
    if(in_model == 2)
    {
        float r2  = x*x + y*y;
        float f = 1 + intrin[4]*r2 + intrin[5]*r2*r2 + intrin[8]*r2*r2*r2;
        float ux = x*f + 2*intrin[6]*x*y + intrin[7]*(r2 + 2*x*x);
        float uy = y*f + 2*intrin[7]*x*y + intrin[6]*(r2 + 2*y*y);
        x = ux;
        y = uy;
    }
    point[0] = depth * x;
    point[1] = depth * y;
    point[2] = depth;
}

/* Transform 3D coordinates relative to one sensor to 3D coordinates relative to another viewpoint */
/* Device function equivalent for the RealSense function rs2_transform_point_to_point */
__device__ void transform_point_to_point(float to_point[3], float extrin[12], float from_point[3]) // extrin: Rotation(0-8), Translation(9-11)
{
    to_point[0] = extrin[0] * from_point[0] + extrin[3] * from_point[1] + extrin[6] * from_point[2] + extrin[9];
    to_point[1] = extrin[1] * from_point[0] + extrin[4] * from_point[1] + extrin[7] * from_point[2] + extrin[10];
    to_point[2] = extrin[2] * from_point[0] + extrin[5] * from_point[1] + extrin[8] * from_point[2] + extrin[11];
}

/* Device function for rounding-off pixel co-ordinates */
__device__ int round_pix_x(float pix_x){
	if (pix_x > 639.5 || pix_x < -0.5) {return -1;}
	return (int)(pix_x - fmod((pix_x+0.5),1.0) + 0.5);
}
__device__ int round_pix_y(float pix_y){
	if (pix_y > 479.5 || pix_y < -0.5) {return -1;}
	return (int)(pix_y - fmod((pix_y+0.5),1.0) + 0.5);
}

/* Global function to be called from Host */
__global__ void transform_d_img(float cu_intrin[18], float cu_extrin[12], int cu_in_model[2], unsigned short cu_depth[640*480], unsigned short cu_tr_depth[640*480], double * depthScale){
	int tid = threadIdx.x; // 0-479
	int bid = blockIdx.x; // 0-639

	if (cu_depth[tid*640+bid] < 0.05) return;

	float f_point[3], t_point[3], t_pixel[2];
	float pixel[2] = {bid, tid};

	deproject_pixel_to_point(f_point, (cu_intrin+9), cu_in_model[1], pixel, (*depthScale)*cu_depth[tid*640+bid]);
	transform_point_to_point(t_point, cu_extrin, f_point);
	project_point_to_pixel(t_pixel, cu_intrin, cu_in_model[0], t_point);
	int x = (int) (t_pixel[0]+0.5f);//round_pix_x(t_pixel[0]);
	int y = (int) (t_pixel[1]+0.5f);//round_pix_y(t_pixel[1]);
	if (x <= -1 || y <= -1 || x >= 639 || y >= 479) return;
	cu_tr_depth[y*640+x] = cu_depth[tid*640+bid];
}


/*								*/
/*				Host			*/
/*								*/

int main(){

	// depth cam config
	std::array<int, 2> depthRes = {640, 480};
	//std::array<int, 2> IRRes    = {640, 480};
	std::array<int, 2> colorRes = {640, 480};
	int colorFPS = 60;
	//int IRFPS    = 90;
	int depthFPS = 90;

	// create rs pipeline
	rs2::pipeline pipe;

	// create configuration
	rs2::config rsCfg;

	rsCfg.enable_stream(RS2_STREAM_COLOR, colorRes[0], colorRes[1], RS2_FORMAT_BGR8, colorFPS);
	rsCfg.enable_stream(RS2_STREAM_DEPTH, depthRes[0], depthRes[1], RS2_FORMAT_Z16, depthFPS);

	// start streaming
	rs2::pipeline_profile profile = pipe.start(rsCfg);
	
	// get color and depth streams
	auto depth_stream = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
	auto color_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();

	// get camera parameters
	double depthScale = profile.get_device().first<rs2::depth_sensor>().get_depth_scale();
	auto c_int = color_stream.get_intrinsics();
	auto d_int = depth_stream.get_intrinsics();
	auto d_to_c_ext = depth_stream.get_extrinsics_to(color_stream);

	// Create new thread
	rs2::frame_queue frameQueue(5);
    std::atomic_bool alive {true};

    /* This thread used solely to receive frames and check if color and depth frames are valid */
    std::thread rxFrame([&]() {
        while(alive) {

            rs2::frameset frames = pipe.wait_for_frames();

            auto cFrame = frames.get_color_frame();
            auto dFrame = frames.get_depth_frame();

            if (!cFrame || !dFrame) {
                continue;
            }
            frameQueue.enqueue(frames);
        }
    });
	
	rs2::frameset curFrame;
    //auto start = std::chrono::high_resolution_clock::now();
    char frameRate[10];


	while(alive) {
        /* Receive frames from other thread here */
        frameQueue.poll_for_frame(&curFrame);

        if (curFrame) {
			auto colorFrame = curFrame.get_color_frame();
			auto depthFrame = curFrame.get_depth_frame();

			// Create Mat from frames
			int color_width  = colorFrame.get_width();
            int color_height = colorFrame.get_height();
            int depth_width  = depthFrame.get_width();
            int depth_height = depthFrame.get_height();

            Mat color(Size(color_width, color_height), CV_8UC3,  (void*)colorFrame.get_data(), Mat::AUTO_STEP);
            Mat depth(Size(depth_width, depth_height), CV_16UC1, (void*)depthFrame.get_data(), Mat::AUTO_STEP);

            auto start = std::chrono::high_resolution_clock::now();

			/* Start kernel for aligning */

			// Allocate memory for cu_intrin, cu_extrin, cu_in_model
			float temp_intrin[18] = {c_int.fx, c_int.fy, c_int.ppx, c_int.ppy, c_int.coeffs[0], c_int.coeffs[1], c_int.coeffs[2], c_int.coeffs[3], c_int.coeffs[4], d_int.fx, d_int.fy, d_int.ppx, d_int.ppy, d_int.coeffs[0], d_int.coeffs[1], d_int.coeffs[2], d_int.coeffs[3], d_int.coeffs[4]};
			float temp_extrin[12] = {d_to_c_ext.rotation[0], d_to_c_ext.rotation[1], d_to_c_ext.rotation[2], d_to_c_ext.rotation[3], d_to_c_ext.rotation[4], d_to_c_ext.rotation[5], d_to_c_ext.rotation[6], d_to_c_ext.rotation[7], d_to_c_ext.rotation[8], d_to_c_ext.translation[0], d_to_c_ext.translation[1], d_to_c_ext.translation[2]};
			int temp_in_model[2] = {c_int.model, d_int.model};
			Mat tr_depth = Mat::zeros(480, 640, CV_16UC1);

			float * cu_intrin, * cu_extrin;
			int * cu_in_model;
			unsigned short * cu_depth, * cu_tr_depth;
			double * cu_depth_scale;

			cudaMalloc((void **)&cu_intrin, 18*sizeof(float));
			cudaMalloc((void **)&cu_extrin, 12*sizeof(float));
			cudaMalloc((void **)&cu_in_model, 2*sizeof(int));
			cudaMalloc((void **)&cu_depth_scale, sizeof(double));

			// Allocate memory for cu_depth, cu_tr_depth in GpuMat
			//cv::cuda::GpuMat cu_depth(depth);
			//cv::cuda::GpuMat cu_tr_depth(cv::Mat::zeros(depth_width, depth_height, CV_16UC1));
			cudaMalloc((void **)&cu_depth, 640*480*sizeof(unsigned short));
			cudaMalloc((void **)&cu_tr_depth, 640*480*sizeof(unsigned short));

			// Initialize cu_intrin, cu_extrin, cu_in_model, cu_depth, cu_tr_depth(= [0])
			cudaMemcpy(cu_intrin, temp_intrin, 18*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(cu_extrin, temp_extrin, 12*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(cu_in_model, temp_in_model, 2*sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(cu_depth_scale, &depthScale, sizeof(double), cudaMemcpyHostToDevice);

			cudaMemcpy(cu_depth, depth.ptr<unsigned short>(0), 640*480*sizeof(unsigned short), cudaMemcpyHostToDevice);

			// Call global function
			transform_d_img<<<640, 480>>>(cu_intrin, cu_extrin, cu_in_model, cu_depth, cu_tr_depth, cu_depth_scale);

			// Copy cu_tr_depth to tr_depth
			//cu_tr_depth.download(tr_depth);
			cudaMemcpy(tr_depth.ptr<unsigned short>(0), cu_tr_depth, 640*480*sizeof(unsigned short), cudaMemcpyDeviceToHost);

			// cudaFree() - cu_intrin, cu_extrin, cu_in_model, cu_depth, cu_tr_depth
			cudaFree(cu_intrin);
			cudaFree(cu_extrin);
			cudaFree(cu_in_model);
			cudaFree(cu_depth);
			cudaFree(cu_tr_depth);
			cudaFree(cu_depth_scale);
			
			/* Parallel code end */

			/* Determine latency in milliseconds between frames */
            auto elapsed = std::chrono::high_resolution_clock::now() - start;
            float milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
            printf("TIME: %02f\n", milliseconds);
            snprintf(frameRate, sizeof(frameRate), "%02f\n", milliseconds);
            putText(color, frameRate, Point(50, 50), FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar::all(255), 3, 3);

            //start = std::chrono::high_resolution_clock::now();

			// display images
			double min, max, tr_min, tr_max;
			cv::minMaxIdx(depth, &min, &max);
			cv::minMaxIdx(tr_depth, &tr_min, &tr_max);
			cv::Mat adjMap, tr_adjMap;
			cv::convertScaleAbs(depth, adjMap, 255 / max);
			cv::convertScaleAbs(tr_depth, tr_adjMap, 255 / tr_max);
			imshow ("Colour Image", color);
			imshow ("Depth Image", adjMap);
			imshow ("Transformed Depth Image", tr_adjMap);
			if (waitKey(1) >= 30){
				destroyAllWindows();
				alive = false;
			}
		}
	}

	rxFrame.join();
	return 0;
}
