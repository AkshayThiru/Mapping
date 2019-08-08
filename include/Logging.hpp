#ifndef LOGGER_H
#define LOGGER_H


#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <unistd.h>
#include <chrono>
#include <time.h>
#include <string>

#include "gnuplot-iostream.h"
#include "Camera.hpp"
#include "Helper.hpp"


//! Logging constants
/*! The follwing boolean values determine the entities to be logged.
*/
///@{

//! If set to true, pose information from T265 will be logged.
static const bool p_logging = true; // logs pose data from T265
//! If set to true, depth intrinsics of D435 will be logged.
static const bool i_logging = true; // logs depth intrinsics data
//! (Not recommended) If set to true, video feed from D435 will be logged.
static const bool v_logging = false; // logs depth feed from D435 (normalized) - use correct depth and video type
//! If set to true, the global map is logged, which can be plotted using cmd 'gnuplot Display.gp'.
static const bool m_logging = true; // logs a point visualization for global map and trajectory
//! (Not recommended) If set to true, a grid visulaization of map is logged.
static const bool g_logging = false; // logs a grid visualization for global map - not recommended
//! (Turn off for performance) If set to true, the depth feed from D$435 is displayed in real-time.
static const bool display = false; // displays depth feed (normalized)
//! (Turn off for performance) If set to true, a 3-D view of the depth feed from D435 is displayed.
static const bool plot_3d = true; // displays 3-D view of depth feed from D435
///@}

//! The path where the log files will be stored.
static const std::string LOG_PATH = "/home/Akshay/Desktop/Test/Mapping/Logs/"; // path for logging



//! Logging class
/*! Instance of this class can be used to log information from the cameras and the global map.
*   Logging can happen either in real-time or after program termination.
*   Real-time logging can cause performance issues, and should be used only for debugging purposes.
*   Correct termination of the program should be ensured in order to avoid inconsistent logged data.
*/
class Logger {

private:

    //! Boolean value to keep track of Logging execution.
	bool start;

    //! High-resolution clock to record timestamps of relevant data.
    std::chrono::high_resolution_clock::time_point ti;

    //! Time at logging initiation.
	time_t today;
    //! Character array to store today.
    char buf[80];
	
	/* output files for logging */
    //! Output log files
    ///@{

    //! Pose log file
	std::ofstream pose_file;
    //! Depth intrinsics file
    std::ofstream d_in_file;
    //! Depth feed video file
    cv::VideoWriter depth_file;
    //! Global map file
    std::ofstream map_file;
    //! Grid file
    std::ofstream grid_file;
    ///@}

    //! Gnuplot instance
    Gnuplot gp;

public:

    //! Default Constructor
    /*! Current day and time are stored into the buf char array.
    */
    Logger () {
    	today = time(0);
    	strftime (buf, sizeof(buf), "%Y_%m_%d_%H_%M_%S", localtime(&today));
    	start = false;
    }

    //! Initializes Logger
    /*! The output files are memory mapped and opened with the corresponding file names.
    */
    void Init() {
    	if (p_logging)
    		pose_file.open(LOG_PATH+"pose.tsv");
    	if (i_logging)
    		d_in_file.open(LOG_PATH+"intrinsics.csv");
    	if (v_logging)
    		depth_file.open(LOG_PATH+std::string(buf)+".avi", CV_FOURCC('F','F','V','1'), INPUT_RATE, cv::Size(w,h), false);
    	if (m_logging)
    		map_file.open(LOG_PATH+"map.tsv");
    	if (g_logging)
    		grid_file.open(LOG_PATH+"grid.gp");
    }

    //! Real-time logging method
    /*! All real-time logging and display are done in this method.
    *   Operations like display video feed or 3-D display can limit performance.
    *   But, it is recommended that pose logging is always set.
    *   \param Camera object
    *   \param Pose from T265
    *   \param 16-bit depth image from D435
    */
    void Log (Camera const * C, rs2_pose const * pose, cv::Mat const * depth) {
        float xl, yu, xr, yd;
        xl = -C->ppx/C->fx*D435_MAX; xr = (w-1-C->ppx)/C->fx*D435_MAX;
        yu = -C->ppy/C->fy*D435_MAX; yd = (h-1-C->ppy)/C->fy*D435_MAX;

        if (!start) {
            ti = std::chrono::high_resolution_clock::now();
            start = true;
            if (plot_3d) {
                gp << "set view 180, 0\n";
                gp << "set xrange ["<<xl<<":"<<xr<<"]\n";
                gp << "set yrange ["<<yu<<":"<<yd<<"]\n";
                gp << "set zrange [0:"<<D435_MAX<<"]\n";
                gp << "set cbrange [0:"<<D435_MAX<<"]\n";
            }
        }

        auto tf = std::chrono::high_resolution_clock::now() - ti;
        double t = std::chrono::duration_cast<std::chrono::milliseconds>(tf).count();
        if (v_logging || display) {
            cv::Mat adj_depth;
            cv::convertScaleAbs(*depth, adj_depth, 255.0/65535.0);
            cv::threshold (adj_depth, adj_depth, D435_MAX/C->scale * 255.0/65535.0, 0, cv::THRESH_TRUNC);
            cv::convertScaleAbs(adj_depth, adj_depth, 65535.0*C->scale/D435_MAX);

            if (v_logging)
                depth_file.write(adj_depth);

            if (display)
                imshow ("Depth Image", adj_depth);
        }
        if (p_logging)
            pose_file << t << " " << pose->translation.x << " " << pose->translation.y << " " << pose->translation.z << " " << pose->rotation.w << " " << pose->rotation.x << " " << pose->rotation.y << " " << pose->rotation.z << " " << pose->tracker_confidence << "\n";

        if (plot_3d) {
            float x_D435, y_D435, z_D435;
            std::vector< std::tuple<float, float, float> > points;
            points.push_back(std::make_tuple(0, 0, 0));
            for (int i = 0; i < h; i+=10) {
                for (int j = 0; j < w; j+=10) {
                    z_D435 = depth->at<unsigned short int>(i,j) * C->scale;
                    x_D435 = (j-C->ppx)/C->fx * z_D435;
                    y_D435 = (i-C->ppy)/C->fy * z_D435;

                    if (z_D435 >= D435_MIN && z_D435 <= D435_MAX)
                        points.push_back(std::make_tuple(x_D435, y_D435, z_D435));
                }
            }
            gp << "set key off\n";
            gp << "set view equal xyz\n";
            gp << "set object polygon from "<<xl<<","<<yu<<","<<D435_MAX<<" to "<<xr<<","<<yu<<","<<D435_MAX<<" to "<<xr<<","<<yd<<","<<D435_MAX<<" to "<<xl<<","<<yd<<","<<D435_MAX<<" to "<<xl<<","<<yu<<","<<D435_MAX<<" fs transparent solid 0 fc rgb 'black' lw 0.1\n";
            gp << "splot '-' using 1:2:3 with points pointsize 0.25 pointtype 8 palette, \\\n";
            gp << "'-' using 1:2:3:($4-$1):($5-$2):($6-$3) with vectors nohead lc rgb 'black' lw 0.25\n";
            gp.send1d(points);
            gp << "0 0 0 "<<xl<<" "<<yu<<" "<<D435_MAX<<"\n";
            gp << "0 0 0 "<<xr<<" "<<yu<<" "<<D435_MAX<<"\n";
            gp << "0 0 0 "<<xr<<" "<<yd<<" "<<D435_MAX<<"\n";
            gp << "0 0 0 "<<xl<<" "<<yd<<" "<<D435_MAX<<"\n";
            gp << "e\n";
            gp << "pause 0.05\n";
        }

    }

    //! Closes the lgging operation.
    /*! All non real-time logging is done in this method.
    *   It also closes the files in memory so that they can accessed later.
    *   Since a pointer to Map_FE object is taken as input, any valid map implementation, inherited from Map_FE will be consistent with the method.
    *   \param Camea object
    *   \param Map_FE pointer
    */
    void Close(Camera const * C, Map_FE * F) {
        if (v_logging)
            depth_file.release();
        if (m_logging) {
            std::vector< std::tuple<float, float, float, float> > points;
            F->Points(&points);
            for (std::vector< std::tuple<float, float, float, float> >::iterator it = points.begin(); it != points.end(); it++) {
                map_file << std::get<0>(*it) << " " << std::get<1>(*it) << " " << std::get<2>(*it) << " " << std::get<3>(*it) << "\n";
            }
            map_file.close();
        }
        if (p_logging)
            pose_file.close();
        if (i_logging) {
            d_in_file << "scale," << C->scale << "\n";
            d_in_file << "focal length," << C->fx << "," << C->fy << "\n";
            d_in_file << "center," << C->ppx << "," << C->ppy << "\n";
            d_in_file << "distortion model," << C->model << "\n";
            d_in_file << "coefficients," << C->coeffs[0] << "," << C->coeffs[1] << "," << C->coeffs[2] << "," << C->coeffs[3] << "," << C->coeffs[4] << "\n";
            d_in_file.close();
        }
        if (g_logging) {
            this->obj_grid(F);
            grid_file.close();
        }

    }

private:

    //! Constructs a grid representation of the map.
    /*! This method creates a gnuplot file, which can run using cmd 'gnuplot <file-name>'.
    *   Logger::point_grid() is called on each of the leaf node points, which are aquired by Map_FE::Points().
    *   Called by Logger::Close()
    *   \param Map_FE pointer
    *   \see Logger::Close(), Logger::point_grid(), Map_FE::Points()
    */
    void obj_grid (Map_FE * F) {
        std::vector< std::tuple<float, float, float, float> > points;
        F->Points(&points);
        grid_file << "set key off\n";
        grid_file << "set xrange [-4:4]\n";
        grid_file << "set yrange [-4:4]\n";
        grid_file << "set zrange [-4:4]\n";
        grid_file << "set view equal xyz\n";

        for (std::vector< std::tuple<float, float, float, float> >::iterator it = points.begin(); it != points.end(); it++) {
            float m_x = fmodf(fmodf(std::get<0>(*it), VOX_L) + VOX_L, VOX_L);
            float m_y = fmodf(fmodf(std::get<1>(*it), VOX_L) + VOX_L, VOX_L);
            float m_z = fmodf(fmodf(std::get<2>(*it), VOX_L) + VOX_L, VOX_L);
            
            this->point_grid (std::get<0>(*it)-m_x, std::get<1>(*it)-m_y, std::get<2>(*it)-m_z, m_x, m_y, m_z, VOX_L);
        }

        grid_file << "splot '-' with points pointsize 0.25 pointtype 7\n";
        grid_file << "0 0 0\n";
        grid_file << "e\n";
        grid_file << "pause -1\n";
    }

    //! Recursively constructs a voxel wireframe
    /*! This method constructs a wireframe around a voxel at each level of the octree.
    *   This is recursive method.
    *   \param Co-ordinate of the origin of this voxel
    *   \param Co-ordinate of the point with respect to the voxel
    *   \param Size of the voxel at the current level
    *   \see Logger::obj_grid()
    */
    void point_grid (float x, float y, float z, float m_x, float m_y, float m_z, float size) {

        grid_file << "set object polygon from "<<x<<","<<y<<","<<z<<" to "<<x+size<<","<<y<<","<<z<<" to "<<x+size<<","<<y+size<<","<<z<<" to "<<x<<","<<y+size<<","<<z<<" to "<<x<<","<<y<<","<<z; if (size/2 < MIN_L) {grid_file << "fs transparent solid 1 fc rgb 'red' lw 0.1\n";} else {grid_file << "fs transparent solid 0 fc rgb 'black' lw 0.1\n";}
        grid_file << "set object polygon from "<<x<<","<<y<<","<<z<<" to "<<x+size<<","<<y<<","<<z<<" to "<<x+size<<","<<y<<","<<z+size<<" to "<<x<<","<<y<<","<<z+size<<" to "<<x<<","<<y<<","<<z; if (size/2 < MIN_L) {grid_file << "fs transparent solid 1 fc rgb 'red' lw 0.1\n";} else {grid_file << "fs transparent solid 0 fc rgb 'black' lw 0.1\n";}
        grid_file << "set object polygon from "<<x<<","<<y<<","<<z<<" to "<<x<<","<<y<<","<<z+size<<" to "<<x<<","<<y+size<<","<<z+size<<" to "<<x<<","<<y+size<<","<<z<<" to "<<x<<","<<y<<","<<z; if (size/2 < MIN_L) {grid_file << "fs transparent solid 1 fc rgb 'red' lw 0.1\n";} else {grid_file << "fs transparent solid 0 fc rgb 'black' lw 0.1\n";}
        grid_file << "set object polygon from "<<x+size<<","<<y+size<<","<<z+size<<" to "<<x<<","<<y+size<<","<<z+size<<" to "<<x<<","<<y<<","<<z+size<<" to "<<x+size<<","<<y<<","<<z+size<<" to "<<x+size<<","<<y+size<<","<<z+size; if (size/2 < MIN_L) {grid_file << "fs transparent solid 1 fc rgb 'red' lw 0.1\n";} else {grid_file << "fs transparent solid 0 fc rgb 'black' lw 0.1\n";}
        grid_file << "set object polygon from "<<x+size<<","<<y+size<<","<<z+size<<" to "<<x<<","<<y+size<<","<<z+size<<" to "<<x<<","<<y+size<<","<<z<<" to "<<x+size<<","<<y+size<<","<<z<<" to "<<x+size<<","<<y+size<<","<<z+size; if (size/2 < MIN_L) {grid_file << "fs transparent solid 1 fc rgb 'red' lw 0.1\n";} else {grid_file << "fs transparent solid 0 fc rgb 'black' lw 0.1\n";}
        grid_file << "set object polygon from "<<x+size<<","<<y+size<<","<<z+size<<" to "<<x+size<<","<<y<<","<<z+size<<" to "<<x+size<<","<<y<<","<<z<<" to "<<x+size<<","<<y+size<<","<<z<<" to "<<x+size<<","<<y+size<<","<<z+size; if (size/2 < MIN_L) {grid_file << "fs transparent solid 1 fc rgb 'red' lw 0.1\n";} else {grid_file << "fs transparent solid 0 fc rgb 'black' lw 0.1\n";}
    
        if (size/2 >= MIN_L)
            this->point_grid (x+m_x-fmodf(m_x,size/2), y+m_y-fmodf(m_y,size/2), z+m_z-fmodf(m_z,size/2), fmodf(m_x,size/2), fmodf(m_y,size/2), fmodf(m_z,size/2), size/2);
    }

};


#endif