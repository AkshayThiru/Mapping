#ifndef CAMERA_H
#define CAMERA_H


#include <librealsense2/rs.hpp>     // Include RealSense Cross Platform API

#include <string>



/*! \name D435: Image Dimensions
*/
///@{
//! Width
static const int w = 640;
//! Height
static const int h = 480;
///@}

static const int d_fps = 90; //!< Frame rate for D435

//@{
//! Minimum and maximum depth for D435
/*! NOTE: don't use D435_MIN less than 0.11 m
*/
static const double D435_MIN = 0.11; // | - min and max depths of D435 (m)
static const double D435_MAX = 2.00; // |
//@}

/*! \name Camera Serial Numbers
*/
///@{
static const std::string DEPTH_SNO = "819612073628"; //!< Serial number of D435
static const std::string TRACK_SNO = "905312110622"; //!< Serial number of T265
///@}

//! Length of Queue Buffers for cameras.
#define BUFFER_LENGTH 5 // length of buffer for cameras
//! Maximum rate at which data receiving thread runs.
/*! \see CPU_main.cpp, GPU_main.cpp
*/
#define INPUT_RATE 50 // Rate at which camera feed is taken (Hz)       | - Maximum rates
//! Rate at which the Global Map is updated.
/*! \see CPU_main.cpp, GPU_main.cpp
*/
#define MAP_UPDATE_RATE 10 // Rate at which global map is updated (Hz) |




//! Struct returned on Camera::Init()
/*! Struct contains two boolean values each denoting whether the corresponding camera stream was started.
*	\see Camera::Init()
*/
struct Bool_Init {
	//! boolean value for T265
	bool t265;
	//! boolean value for D435
	bool d435;
};

//! Camera streams abstraction class
/*!
*	This class is used to initialize the D435 - Depth camera, and T265 - Tracking camera.
*	The class object can either be used directly, or used along with Cam_RW.hpp as a publisher.
*	All device properties can be modified in this class.
*	\see Cam_RW.hpp
*/
class Camera {

private:

	//! Realsense context object
	/*! The members of this object can be set and passed to rs2::pipeline constructor to set the properties of the cameras.
	*/
	rs2::context ctx;

public:

	//! Used to call wait_for_frames()
	/*! Elements of this vector can be used to wait for frames.
	*	If only camera is attached, the vector contains only one element.
	*	If both cameras are attached, the first element is for T265 and the second for D435
	*/
	std::vector<rs2::pipeline> pipelines; // pipelines for depth and tracking cameras - 0: tracking, 1: depth

	/*! \name D435 Intrinsics
	*	Depth camera properties
	*/
	///@{
	float scale; //!< Depth scale (m)
	//@{
	//! Focal length: x (pixels)
	float fx;
	//! Focal length: y (pixels)
	float fy;
	//@}
	//@{
	//! Image center: x (pixels)
	float ppx;
	//! Image center: y (pixels)
	float ppy;
	//@}
	//! Distortion model type
	int model;
	//! Distortion Coefficients
	float coeffs[5];
	///@}

	/*! @name Frame Queue 
	*	Frame queues for tracking and depth 
	*/
	///@{
	//! Queue for Depth frames
	rs2::frame_queue d_queue;
	/*! Queue for Pose frames */
	rs2::frame_queue t_queue;
	///@}

	//! Default Constructor
	/*! Initializes the Queues with a size of BUFFER_LENGTH 
	*	@see BUFFER_LENGTH
	*/ 
	Camera(): d_queue(BUFFER_LENGTH), t_queue(BUFFER_LENGTH) {}

	//! Initialize and start camera streams
	/*! Properties of the streams are set in this method. <br>
	*	D435: Currently only Depth image is streamed. Image dimensions, bit depth, and FPS of D435 can be set in this method. <br>
	*	T265: Currently only 6-DoF Pose is streamed. The Degrees of Freedom of Pose can be set in this method.
	*	run rs-enumerate-devices in terminal to view available configurations <br>
	*	NOTE: The serial number is different for every camera (even for the same %model). This param should be set for every new device.
	*	\see w, h, d_fps, DEPTH_SNO, TRACK_SNO, Bool_Init
	*	\return a Bool_Init struct stating which cameras where initialzed.
	*/
	Bool_Init Init () try {
		int num_dev = 0;
		std::vector<rs2::pipeline> temp;
		int d_idx, t_idx;
		Bool_Init b {false, false};

		for (auto&& dev : ctx.query_devices())
	    {

	        rs2::pipeline pipe(ctx);
	        rs2::config cfg;
	        cfg.enable_device(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
	        
	        if (strcmp(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER), &DEPTH_SNO[0]) == 0) {
	        	cfg.enable_stream(RS2_STREAM_DEPTH, w, h, RS2_FORMAT_Z16, d_fps);
	        	std::cout << "Depth Camera initialized: {" << w << "," << h << "}, 90 FPS\n";

	            rs2::pipeline_profile profile = pipe.start(cfg);
	            auto stream = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
	            scale = profile.get_device().first<rs2::depth_sensor>().get_depth_scale();
	            auto intrinsics = stream.get_intrinsics();
	            fx  = intrinsics.fx;
	            fy  = intrinsics.fy;
	            ppx = intrinsics.ppx;
	            ppy = intrinsics.ppy;
	            model = intrinsics.model;
	            for (int i = 0; i < 5; i++)
	            	coeffs[i] = intrinsics.coeffs[i];

	            d_idx = num_dev;
	            b.d435 = true;
	        }
	        else if (strcmp(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER), &TRACK_SNO[0]) == 0) {
	        	cfg.enable_stream(RS2_STREAM_POSE, RS2_FORMAT_6DOF);
	        	std::cout << "Tracking Camera initialized: 6DoF\n";

	            pipe.start(cfg);
	            t_idx = num_dev;
	            b.t265 = true;
	        }
	        else {
	        	std::cout << "Device not recognized. Serial Number: " << dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) << "\n";
	        	return b;
	        }

	        temp.emplace_back(pipe);
	        num_dev++;
	    }

	    if (b.t265)
	    	pipelines.emplace_back(temp[t_idx]);
	    if (b.d435)
	    	pipelines.emplace_back(temp[d_idx]);

	    return b;

	}
	catch (const rs2::error & e) {
		std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
		return Bool_Init {false, false};
	}
	catch (const std::exception & e) {
		std::cerr << e.what() << std::endl;
		return Bool_Init {false, false};
	}

};


#endif