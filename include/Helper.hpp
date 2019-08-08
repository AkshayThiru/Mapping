#ifndef HELPER_H
#define HELPER_H

#ifdef __CUDACC__
#define CUDA_CALL __host__ __device__
#else
#define CUDA_CALL
#endif


#include "Camera.hpp"
#include <opencv2/opencv.hpp>




/* Use following template class as replacement for std::pair */
//! Template Class for Pairs.
/*! This class is used as a replacement for std::pair for CUDA code.
*	Note that STL classes and methods should preferably not used in CUDA as they might cause memory access errors.
*	The '<' operator is defined on %Pair.first, so '<' should be defined for template class A.
*	This is used to sort a vector of this template class in CUDA as a replacement for std::map - which uses a red-black tree implementation.
*	\see GPU_FE::Update()
*/
template <typename A, typename B>
class Pair {

public:

	//! First member of Pair. Can be used as an index.
	A first;
	//! Second member of Pair. Can be used as mapped value.
	B second;

	/*! \name Constructors
	*/
	///@{

	//! Default Constructor
	CUDA_CALL Pair () = default;

	//! Equivalent to Pair(), Pair.A(a), Pair.B(b)
	/*!	\param object of type A
	*	\param object of type B
	*/
	CUDA_CALL Pair (const A a, const B b): first(a), second(b) { }
	///@}

	/*! \name Operator Overrides
	*/
	///@{
	
	//! overridding of '<' operator.
	/*! Used to sort vectors of element type Pair<A,B>.
	*	\param Pair P of types A, and B
	*	\return boolean value comparing the first elements
	*	\see GPU_FE::Update()
	*/
	CUDA_CALL inline bool operator < (Pair<A, B> const &P) const {
		return (this->first < P.first);
	}

	//! overridding of '==' operator.
	CUDA_CALL inline bool operator == (Pair<A, B> const &P) const {
		return (this->first == P.first);
	}
	///@}

};


//! Virtual class Parent of CPU_FE and GPU_FE classes.
/*! Classes CPU_FE and GPU_FE are the front-end classes for CPU and GPU versions of the algorithm respectively.
*	Both these classes inherit MAP_FE, which is a virtual class:
*	meaning an object of this class cannot be created. But such an implementation ensures two things: <br>
*	1.  Any implementation of any mapping algorithm must neccessarily implement the member methods of MAP_FE. <br>
*	2.  Other classes dependent on the global map need not change their implementation depending on the algorithm used as long
*		as the front end of the implementation is a child of Map_FE.
*	Also pointer of a child class can be cast to their parent class.
*	\see CPU_FE, GPU_FE
*/
class Map_FE {

public:

	//! Method to update global map
	/*! This method runs at every iteration of frame recieved at a rate of MAP_UPDATE_RATE.
	*	\param Camera object
	*	\param current pose from T265
	*	\param 16-bit (default) depth image from D435
	*/
	virtual void Update (Camera const &C, rs2_pose const &pose, cv::Mat const &depth) = 0;

	//! Returns all points in the map
	/*! Primarily to be used by Logger class. The points returned are in no particular order.
	*	\param vector of tuple containing (x, y, z)
	*	\param variance of points
	*	\see Logger::Log()
	*/
	virtual void Points (std::vector < std::tuple<float, float, float, float> > * points) = 0;

};


#endif