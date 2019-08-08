#ifndef VOXEL_CH
#define VOXEL_CH


#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <iostream>
#include <fstream>

#include <typeinfo>
#include <cmath>
#include <math.h>
#include <utility>
#include <unistd.h>
#include <thrust/tuple.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>

#include "Helper.hpp"

//! Minimum dimension of leaf node.
/*! The Voxels will keep dividing until their the size of voxel is \f$\leq\f$ MIN_L, at which point a leaf is alloted in place of a voxel.
*	Set the value as a floating value. eg: 1.00
*/
#define MIN_L 0.04 // minimum edge length of leaf in meter
//! Size of root voxels.
/*! The starting size of root voxels. This should not be \f$\leq\f$ MIN_L.
*	Set the value as a floating value. eg: 3.00
*/
#define VOX_L 2.56 // edge length of root voxel in meter (>= MIN_L). Define as float:eg: 2.00
//! Variance of measurement
/*! This is the 3-D variance of each point measured. Assumed constant and isotropic.
*	The co-variance matrix in this case is \f$ VAR\_P . \mathbb{1}_{3{\times}3} \f$
*/
#define VAR_P 0.005 // variance of each measurement
//! \name Kernel Launch Parameters
/*! Note: Launch parameters should satisfy all constraints. Run deviceQuery in CUDA samples to check device characteristics.
*/
///@{

//! Number of blocks launched in the grid
/*! Should be less than maximum Grid size in all dimensions
*/
#define NUM_B 480
//! Number of threads launched in each block
/*! Should be less than maximum Block size in all dimensions
*/
#define NUM_T 640
///@}


//! A basic Quaternion class
/*! GPU: <br>
*	Quaternion class with components \f$ x, y, z, w \f$ such that \f$ q = x\textit{i} + y\textit{j} + z\textit{k} + w\f$.
*	Basic operators provided are \f$\times\f$: multiplication and \f$+\f$: addition. \f$^{-1}\f$: inverse is provided through quaternion::inv() method.
*	Can be used in both host and device code.
*/
class quaternion {

public:

	//! \name Components of quaternion.
	///@{
	float x, y, z, w;
	///@}

	//! Constructor taking x, y, z, w in order
	/*! Note: This is the only constructor provided.
	*	Can be used in both host and device.
	*	\param Components: \f$\textit{i}\f$, \f$\textit{j}\f$, \f$\textit{k}\f$, and \f$\mathbb{R}\f$
	*/
	__host__ __device__ quaternion (float x, float y, float z, float w) {
		this->x = x;
		this->y = y;
		this->z = z;
		this->w = w;
	}

	//! Inverse of the quaternion
	/*! To be used as q.inv() \f$\equiv q^{-1}\f$ 
	*	\return quaternion
	*/
	__host__ __device__ quaternion inv () {
		float l = (this->x)*(this->x) + (this->y)*(this->y) + (this->z)*(this->z) + (this->w)*(this->w);
		return quaternion (-(this->x)/l, -(this->y)/l, -(this->z)/l, (this->w)/l);
	}

	//! \f$\times\f$ operator
	/*	To be used as \f$q_1*q_2\f$ \f$\equiv q_1\timesq_2\f$, where &\f$q_1\f$ = this
	*	\param \f$q_2\f$
	*	\return quaternion
	*/
	__host__ __device__ quaternion operator * (quaternion const &q) {
		quaternion q_t(0, 0, 0, 0);
		q_t.x = + this->x*q.w + this->y*q.z - this->z*q.y + this->w*q.x;
		q_t.y = - this->x*q.z + this->y*q.w + this->z*q.x + this->w*q.y;
		q_t.z = + this->x*q.y - this->y*q.x + this->z*q.w + this->w*q.z;
		q_t.w = - this->x*q.x - this->y*q.y - this->z*q.z + this->w*q.w;
		return q_t;
	}

	//! \f$+\f$ operator
	/*	To be used as \f$q_1+q_2\f$ \f$\equiv q_1+q_2\f$, where &\f$q_1\f$ = this
	*	\param \f$q_2\f$
	*	\return quaternion
	*/
	__host__ __device__ quaternion operator + (quaternion const &q) {
		return quaternion (this->x+q.x, this->y+q.y, this->z+q.z, this->w+q.w);
	}
};

//! Pose of T265 camera
/*! Used to pass to CUDA kernel
*/
struct Pose {
	//! Translation of T265 expressed as a quaternion in T265 frame
	quaternion t;
	//! Rotation of T265 expressed as a quaternion in T265 frame
	quaternion r;
};

//! Camera Intrinsics and Extrinsics
/*!	Used to pass to CUDA kernel
*/
struct Cam {

	//! \name Focal length (pixels)
	///@{

	float fx, fy;   // |
	///@}

	//! \name Image Center (pixels)
	///@{

	float ppx, ppy; // | - Camera intrinsics
	///@}

	//! Depth scale (\f$ \textit{m}\f$)
	float scale;    // |

	//! \name T265 to D435 Extrinsics
	///@{

	quaternion Q_TD, T_TD; // Camera extrinsics
	///@}
};

//! %Point co-ordinates and variance
/*! Used to pass to CUDA kernel
*/
struct Tuple {

	//! \name Point co-ordinates
	///@{

	float x, y, z;
	///@}

	//! \name Variance
	float c;
};

//! Point co-ordinates
/*! Used to pass to CUDA kernel
*/
struct Point {

	//! \name Point co-ordinates
	///@{

	float x, y, z;
	///@}
};


//! \name T265 to D435 extrinsics
/*! To be obtained from extrinsic calibration data of the mount.
*/
///@{

//! Quaternion from \f$\mathfrak{R}_{T265} \to \mathfrak{R}_{D435}\f$ in \f$\mathfrak{R}_{T265}\f$
static const quaternion Q_T265_D435 (-0.0089999, 0.0024999, 0.0000225, 0.9999564); // | - T265 to D435 extrinsics
//! Translation from \f$\mathfrak{R}_{T265} \to \mathfrak{R}_{D435}\f$ in \f$\mathfrak{R}_{T265} (m)\f$
static const quaternion T_T265_D435 (0.021, 0.027, 0.009, 0);                      // |
///@}



//! Prints out errors in CUDA kernel execution
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if( abort )
            exit(code);
    }
}

//! Method to print out errors in CUDA kernel execution
inline void gpuCheckKernelExecutionError( const char *file, int line)
{
    gpuAssert( cudaPeekAtLastError(), file, line);
    gpuAssert( cudaDeviceSynchronize(), file, line);    
}


/* leaf class */
//! Leaf nodes of the Octree structure
/*! GPU: <br>
*	This is not implemented as a voxel object because there can be millions of nodes and so the size should be as small as possible.
*	Stores the x, y, z co-ordinates of a single point inside it relative to edge length ie. \f$x, y, z \in [0,1)\f$. This is to maintain uniform accuracy
*	across all points. (accuracy of float type reduces as one moves away from 0)
*	The origin of the node is the vertex with all co-ordinates minimum. ie. if the origin of voxel is
*	\f$(x_o, y_o, z_o)\f$ and edge length is \f$L\f$, The vertices of the node are \f$\{(x_o, y_o, z_o), ..., (x_o+L, y_o+L, z_o+L)\}\f$
*	If the member leaf::_v \f$> 0\f$, the leaf node is occupied. If leaf::_v \f$= 0\f$, the leaf node is empty (this is not the same as unobserved. This means that
*	this node has been observed, but there is no point inside it). This has been used becuase if initially a node was observed to be empty, and containing a point afterwards, the same update
*	rule can be used without any change, in a single atomic operation. Although this is not particularly important for the CPU operation, it is extremely essential for the 
*	GPU operation to maintain consistency.
*	An object of this class can only be declared inside the CUDA kernel.
*/
class leaf {

public:

	//! Inverse of variance
	/*! The points are assumed to be distributed as a 3-D uniform gaussian distribution when measured.
	*	As more points are updated in the node, this variance decreases, ie. the certainity of a point existing in the node increases.
	*	The update rule is the typical update rule of gaussian distribution, same as the one in Measurement Update Step in EKF and SLAM.
	*	Inverse of variance is stored so that the update can be performed in a single atomic step while running in GPU.
	*	\see Voxel.cuh
	*/
	float _v;

	//! Co-ordinates of point inside leaf node divided by the variance.
	/*! \name Co-ordinates
	*	The co-ordinates are measured relative to leaf node edge length, ie. \f$x, y, z \in [0,1)\f$.
	*	Note that although x_v, y_v, and z_v can are unbounded, the values of x, y, and z are bounded since the update is a convex combination of two points inside the node.
	*	The co-ordinates are divided by the variance so that the update can be performed in a single atomic operation while running in GPU.
	*	\see Voxel.cuh
	*/
	///@{

	float x_v, y_v, z_v;
	///@}

	//! Constructor for leaf node
	/*! Note that this is the only constructor provided.
	*	\param (x, y, z) relative to leaf node, ie. \f$x, y, z \in [0,1)\f$ for correct operation
	*/
	__device__ leaf (float x, float y, float z) { // state = -1: unoccupied
		_v = 0;
		x_v = y_v = z_v = 0;
	}

	//! Update method for this node object.
	/*! Since every node contains only a single point, this update rule is used to combine the points into a single point.
	*	This is the same as the Measurement Update Step in EKF and SLAM. In this particular case the rule is a simple weighted average.
	*	So, if the point already existing in the node has a very low variance, the updated point will be very close to the previous point.
	*	Even if an anisotropic gaussian probability distribution function is used, the updated point will always be a convex combination of two points.
	*	atommicAdd() function and the transformed variables ensure consistency while multi-threading.
	*	\param (x, y, z) relative to leaf node, ie. \f$x, y, z \in [0,1)\f$ for correct operation
	*/
	__device__ void update_leaf (float x, float y, float z) { // x, y, z: scaled wrt to this->size
		atomicAdd(&x_v, x/VAR_P);
		atomicAdd(&y_v, y/VAR_P);
		atomicAdd(&z_v, z/VAR_P);
		atomicAdd(&_v, 1/VAR_P);
	}

};


/* voxel class */
//! Voxel/Intermediate nodes of the Octree structure
/*! GPU: <br>
*	Primarily stores the pointers to the eight children of this voxel object. Additionally it also stores the co-ordinate of a combined
*	single point, calculated from all its children. This information can be used if memory consumed by the Octree structure reaches a threshold,
*	in which case all the children of a voxel object at some particular level can deleted freeing some space, but at the same time not losing information
*	about the space inside (although accuracy will decrease).
*	The x, y, z co-ordinates of thr single point stored inside is relative to edge length ie. \f$x, y, z \in [0,1)\f$. This is to maintain uniform accuracy
*	across all points. (accuracy of float type reduces as one moves away from 0)
*	The origin of the node is the vertex with all co-ordinates minimum. ie. if the origin of voxel is
*	\f$(x_o, y_o, z_o)\f$ and edge length is \f$L\f$, The vertices of the node are \f$\{(x_o, y_o, z_o), ..., (x_o+L, y_o+L, z_o+L)\}\f$
*	If the member voxel::_v \f$> 0\f$, the leaf node is occupied. If _v \f$= 0\f$, the voxel node is empty (this is not the same as unobserved. This means that
*	this node has been observed, but there is no point inside it). This has been used becuase if initially a node was observed to be empty, and containing a point afterwards, the same update
*	rule can be used without any change, in a single atomic operation.
*	Additionally, if any child pointer c[i] \f$= NULL\f$, then that child has not yet been observed.
*	An object of this class can only be declared inside the CUDA kernel.
*/
class voxel {

public:

	//! Pointers to child voxels/leafs
	/*! The pointers are of type void * becuase the child can either be a voxel node or a leaf node depending on the level, MIN_L, and VOX_L.
	*	The order of numbering is such that the index of smaller co-ordinate child \f$<\f$ index of larger co-ordinate child with the preference among dimensions being \f$ z > y > x\f$
	*	ie. index \f$ = (z\geq0.5)\ll2 \lor (y\geq0.5)\ll1 \lor (x\geq0.5)\f$
	*/
	void * c[8]; // child voxels
	//! Inverse of variance
	/*! The points are assumed to be distributed as a 3-D uniform gaussian distribution when measured.
	*	As more points are updated in the node, this variance decreases, ie. the certainity of a point existing in the node increases.
	*	The update rule is the typical update rule of gaussian distribution, same as the one in Measurement Update Step in EKF and SLAM.
	*	Inverse of variance is stored so that the update can be performed in a single atomic step while running in GPU.
	*/
	float _v; // inverse of variance

	//! Co-ordinates of a single point inside voxel node divided by the variance.
	/*! \name Co-ordinates
	*	The co-ordinates are measured relative to voxel node edge length, ie. \f$x, y, z \in [0,1)\f$.
	*	Note that although x_v, y_v, and z_v can are unbounded, the values of x, y, and z are bounded since the update is a convex combination of two points inside the node.
	*	The co-ordinates are divided by the variance so that the update can be performed in a single atomic operation while running in GPU.
	*/
	///@{

	float x_v, y_v, z_v; // point co-ordinate wrt voxel (0-1) / variance
	///@}
	//! Edge length of voxel node (\f$\textit{m}\f$)
	float size; // edge length of voxel in meter
	/* voxel * p; // parent voxel - initialize in constructor if used */

	//! Constructor for voxel node
	/*! Note that this is the only constructor provided.
	*	\param (x, y, z) relative to node, ie. \f$x, y, z \in [0,1)\f$ for correct operation
	*	\param edge length of voxel (\f$\textit{m}\f$)
	*/
	__device__ voxel (float x, float y, float z, float size) { // state = -1: unoccupied
		_v = 0;
		x_v = y_v = z_v = 0;
		this->size = size;
		c[0] = c[1] = c[2] = c[3] = c[4] = c[5] = c[6] = c[7] = NULL;
	}

	//! Update method for this node object.
	/*! For each voxel, two update steps are performed: one for the child voxel/leaf the input point lies in, and one for this voxel object.
	*	For the child update, it is first checked whether the child exists. If it does, leaf::update_leaf() or voxel::update_vox() is called on the child object.
	*	If it doesn't, a new child voxel/leaf is created and the constructor leaf::leaf() or voxel::voxel() is called.
	*	This step is a recursive one.
	*	To avoid multiple threads creating inconsistent and wasteful copies of the same child node, the following strategy is used:
	*	Each thread creates a copy of child voxel, then an atomic Compare and Swap (atomicCAS()) is applied on the child pointer.
	*	Only one thread can successfully replace the pointer. This pointer is subsequently used for all updates, and the unused children are deleted.
	*	The decision of whether the child is a voxel node or a leaf node is made considering the edge lengths of the children. (\f$=\frac{this\to\_v}{2}\f$)
	*	If child edge length \f$ \leq \f$ MIN_L, the child is a leaf node, else it is a voxel node.
	*	The next step is self update which is similar to leaf::update_leaf()
	*	\param (x, y, z) relative to node, ie. \f$x, y, z \in [0,1)\f$ for correct operation
	*	\see leaf::update_leaf(), voxel::update_self()
	*/
	__device__ void update_vox (float x, float y, float z) { // x, y, z: scaled wrt to this->size

		/* update child voxels */
		int idx = (z >= 0.5)<<2 | (y >= 0.5)<<1 | (x >= 0.5); // idx of child voxel the point lies in

		if (size/4 >= MIN_L) { /* child is a voxel object */
			if (c[idx] == NULL) {
				void * cptr = (void *) new voxel (fmodf(x,0.5)*2, fmodf(y,0.5)*2, fmodf(z,0.5)*2, size/2);
				if ((void *)atomicCAS ((unsigned long long int *)&c[idx], (unsigned long long int)NULL, (unsigned long long int)cptr) != NULL) // child created by some other thread
					delete ((voxel *)cptr);
				((voxel *)c[idx])->update_self(fmodf(x,0.5)*2, fmodf(y,0.5)*2, fmodf(z,0.5)*2);
				((voxel *)c[idx])->update_vox(fmodf(x,0.5)*2, fmodf(y,0.5)*2, fmodf(z,0.5)*2);
			}
			else {
				((voxel *)c[idx])->update_self(fmodf(x,0.5)*2, fmodf(y,0.5)*2, fmodf(z,0.5)*2);
				((voxel *)c[idx])->update_vox (fmodf(x,0.5)*2, fmodf(y,0.5)*2, fmodf(z,0.5)*2);
			}
		}
		else { /* child is a leaf object */
			if (c[idx] == NULL) {
				void * cptr = (void *) new leaf (fmodf(x,0.5)*2, fmodf(y,0.5)*2, fmodf(z,0.5)*2);
				if ((void *)atomicCAS ((unsigned long long int *)&c[idx], (unsigned long long int)NULL, (unsigned long long int)cptr) != NULL) // child created by some other thread
					delete ((leaf *)cptr);
				((leaf *)c[idx])->update_leaf (fmodf(x,0.5)*2, fmodf(y,0.5)*2, fmodf(z,0.5)*2);
			}
			else {
				((leaf *)c[idx])->update_leaf (fmodf(x,0.5)*2, fmodf(y,0.5)*2, fmodf(z,0.5)*2);
			}
		}

	}

	//! Update method for self
	/*! Following the update of the children, the point stored inside this voxel is updated.
	*	atommicAdd() function and the transformed variables ensure consistency while multi-threading.
	*	This method is similar to leaf::update_leaf()
	*	\param (x, y, z) relative to node, ie. \f$x, y, z \in [0,1)\f$ for correct operation
	*	\see leaf::update_leaf(), voxel::update_vox()
	*/
	__device__ void update_self (float x, float y, float z) {
		/* update self */
		atomicAdd(&x_v, x/VAR_P);
		atomicAdd(&y_v, y/VAR_P);
		atomicAdd(&z_v, z/VAR_P);
		atomicAdd(&_v, 1/VAR_P);
	}

	//! Recursively frees up memory inside this voxel node.
	/*! This is called upon by the global method Delete() (which is inturn called by GPU_FE::~GPU_FE()) on each of the root voxel nodes, which recursively deletes all the nodes in the octree.
	*	Run by a single CUDA thread, since it is called only once and doesn't affect the performance.
	*	\see GPU_FE::~GPU_FE(), Delete()
	*/
	__device__ void free_mem () {
		if (size/4 >= MIN_L) { /* child is a voxel object */
			for (int i = 0; i < 8; i++) {
				if (c[i] != NULL) {
					((voxel *)c[i])->free_mem();
					delete (voxel *)c[i];
				}
			}
		}
		else { /* child is a leaf object */
			for (int i = 0; i < 8; i++) {
				if (c[i] != NULL)
					delete (leaf *)c[i];
			}
		}
	}

	//! Appends all leaf node points in this node to vector set.
	/*	This is called by Print() (inturn called by GPU_FE::Points(), which can be user called or called by Logger::Close()) on each
	*	root voxel node, which recursively appends all points to the vector set.
	*	Run by a single CUDA thread, since it is called only once and doesn't affect the performance.
	*	\param co-ordinates of points
	*	\param origin of the voxel node.
	*	\see Print(), GPU_FE::Points(), Logger::Close()
	*/
	__device__ void all_points (Tuple * set, float x_o, float y_o, float z_o, int * idx) {
		if (size/4 >= MIN_L) { /* child is a voxel object */
			for (int i = 0; i < 8; i++) {
				if (c[i] != NULL) {
					((voxel *)c[i])->all_points(set, x_o+size/2*(i&1), y_o+size/2*((i&2)>>1), z_o+size/2*((i&4)>>2), idx);
				}
			}
		}
		else { /* child is a leaf object */
			leaf * p = NULL;
			for (int i = 0; i < 8; i++) {
				if (c[i] != NULL) {
					p = (leaf *) c[i];
					Tuple temp = {x_o+((p->x_v)/(p->_v)+(i&1))*size/2, y_o+((p->y_v)/(p->_v)+((i&2)>>1))*size/2, z_o+((p->z_v)/(p->_v)+((i&4)>>2))*size/2, 1/(p->_v)};
					set[(*idx)++] = temp;
				}
			}
		}
	}

	//! Checks if this node has been observed or not.
	/*! If the node has atleast one filled or empty children, this method returns false.
	*	\see voxel
	*/
	__device__ bool is_empty () {
		for (int i = 0; i < 8; i++) {
			if (c[i] != NULL)
				return false;
		}
		return true;
	}

};


//! Binary search for key in sorted array
/*!	Pointer to a sorted vector (stored in device) is passed along with the size and the starting index, and the binary search algorithm is used to index via key.
*	It is a recursive method.
*	\param Pointer to sorted vector v, beginning index b, ending index e
*	\param key ot index into vector
*	\return Pair of index and voxel with the given index
*	\see Update_root()
*/
__device__ Pair< long, Pair<voxel *, Point> > binary_search (Pair< long, Pair<voxel *, Point> > * v, long b, long e, long key) { // ascending order is assumed
	if (e >= b) {
		long m = b + (e-b)/2;
		if (v[m].first == key)
			return v[m];
		else if (v[m].first > key)
			return binary_search (v, b, m-1, key);
		else 
			return binary_search (v, m+1, e, key);
	}
	return Pair< long, Pair<voxel *, Point> > (0l, Pair<voxel *, Point> (NULL, Point {0, 0, 0}));

}

//! Calculates co-ordinate of point modulo edge length.
/*! Returns \f$p mod VOX\_L[0, 1)^3\f$
*	\param co-ordinate of point
*	\return modulo of co-ordinate of point
*	\see occ_grid::mod_p()
*/
__device__ Point mod_p (Point p) {
		return Point {fmodf(fmodf(p.x, VOX_L) + VOX_L, VOX_L), fmodf(fmodf(p.y, VOX_L) + VOX_L, VOX_L), fmodf(fmodf(p.z, VOX_L) + VOX_L, VOX_L)};
}

//! Calculates index used as key to index into device vector.
/*! This is used to calculate a unique whole number from a set of three integers: indices of origin of the voxel.
*	Instead of using three nested maps each trying to index one co-ordinate at each level (\f$ O(\ln(N_x)+\ln(N_y)+\ln(N_z))\f$), 
*	a bijective mapping from \f$ \mathbb{Z}^{3} \to \mathbb{N}\f$ is defined. Although the order of the complexity remains the same, the look-up is guaranteed to occur in less time than the previous case.
*	\param co-ordinates of the origin of voxel
*	\return index of point
*	\see occ_grid::index(), GPU_FE::Update()
*/
__device__ long index (Point p) {
	Point mod = mod_p(p);
	long a = (p.x < 0) ? -2*std::round((p.x-mod.x)/VOX_L)-1 : 2*std::round((p.x-mod.x)/VOX_L);
	long b = (p.y < 0) ? -2*std::round((p.y-mod.y)/VOX_L)-1 : 2*std::round((p.y-mod.y)/VOX_L);
	long c = (p.z < 0) ? -2*std::round((p.z-mod.z)/VOX_L)-1 : 2*std::round((p.z-mod.z)/VOX_L);
	long idx = (a+b+c+2)*(a+b+c+1)*(a+b+c)/6 + (a+b+1)*(a+b)/2 + a;
	return idx;
}

//! Updates point in the global map.
/*! This method recursively calls voxel::update_vox() on multiple threads concurrently, to update the point in the respective voxel. This GPU kernel itself is called upon by GPU_FE::Update().
*	The information on the origin of the voxel is used to identify the voxel, and the index is used as a key to search in the sorted device vector.
*	This method is the same as voxel::update_vox(), other than the fact that the point doesn't directly map to any "child" voxel.
*	The co-ordinates are transformed from the D435 frame to T265 global frame and then passed on to occ_grid::update_point().
*	Equivalent to occ_grid::update_point(), and CPU_FE::Update().
*	Since inserts and searches into the device vector would have to be done atomically, a temporary array of voxel pointers is used.
*	The size of the array is fixed, and is calculated using D435 intrinsics, D435_MAX, and VOX_L, such that a mapping from each point to the array index can be made.
*	Therefore, every point belonging to the same voxel is mapped to the same index in the array, which can be known.
*	This not only solves the problem of consistency, but also results in almost maximum possible parallel efficiency.
*	This temporary array is appended to the device vector containing root voxels, and is sorted (GPU_FE::Update()).
*	Although sorting a vector, which is a linear array, takes \f$O(n)\f$ as opposed to the \f$O(\ln(n))\f$ for insertion in a map, which is a red-black tree, since new voxels are sparsely created, it is not expected to reduce performance noticeably.
*	This difference in insertion times can be attributed ot the fact that indexing in a linear array is \f$O(1)\f$.
*	\param Camera object
*	\param pose of T265
*	\param 16-bit D435 depth image
*	\param device vector containing root voxel pointers
*	\param size of device vector
*	\see voxel::update_vox(), GPU_FE::Update()
*/
__global__ void Update_root (unsigned short d[w*h], Pair< long, Pair<voxel *, Point> > * v, long * s, Pair< long, Pair<voxel *, Point> > * temp, Cam * c, Pose * p) {
	int tid = threadIdx.x; // 0-(w-1)
	int bid = blockIdx.x; // 0-(h-1)
	int id  = (blockDim.x)*bid+tid;

	for (int i = id; i < w*h; i+=NUM_T*NUM_B) {
	    float z_D435 = d[i] * c->scale;
	    float x_D435 = ((i%w)-c->ppx)/c->fx * z_D435;
	    float y_D435 = ((i/w)-c->ppy)/c->fy * z_D435;

	    quaternion pose_pix = p->t + p->r * quaternion(x_D435,y_D435,z_D435,0) * p->r.inv();

	    if (z_D435 < D435_MIN || z_D435 > D435_MAX)
	    	continue;

	    long idx = index (Point {pose_pix.x, pose_pix.y, pose_pix.z});
	    Point mod = mod_p(Point {pose_pix.x, pose_pix.y, pose_pix.z});
	    Pair< long, Pair<voxel *, Point> > p_idx = binary_search (v, 0l, (*s)-1l, idx);
	    if (p_idx.second.first != NULL) { // voxel containing point exists
	    	((voxel*)p_idx.second.first)->update_self (mod.x/VOX_L, mod.y/VOX_L, mod.z/VOX_L);
	    	p_idx.second.first->update_vox (mod.x/VOX_L, mod.y/VOX_L, mod.z/VOX_L);
	    }
	    else { // voxel doesn't exist
	    	long n1 = std::round((pose_pix.x-mod.x)/VOX_L);
			long n2 = std::round((pose_pix.y-mod.y)/VOX_L);
			long n3 = std::round((pose_pix.z-mod.z)/VOX_L);

			void * cptr = (void *)new voxel (mod.x/VOX_L, mod.y/VOX_L, mod.z/VOX_L, VOX_L);
			void * ac_ptr = (void *)atomicCAS ((unsigned long long int *)&(temp[25*(((n3%5)+5)%5)+5*(((n2%5)+5)%5)+(((n1%5)+5)%5)].second.first), (unsigned long long int)NULL, (unsigned long long int)cptr);
			if (ac_ptr != NULL) {// voxel created by some other thread
				delete((voxel *)cptr);
				((voxel *)ac_ptr)->update_self (mod.x/VOX_L, mod.y/VOX_L, mod.z/VOX_L);
				((voxel *)ac_ptr)->update_vox (mod.x/VOX_L, mod.y/VOX_L, mod.z/VOX_L);
			}
			else { // voxel created by current thread
				Pair< long, Pair<voxel *, Point> > p_temp(idx, Pair<voxel *, Point>((voxel *)cptr, Point {pose_pix.x-mod.x, pose_pix.y-mod.y, pose_pix.z-mod.z}));
				temp[25*(((n3%5)+5)%5)+5*(((n2%5)+5)%5)+(((n1%5)+5)%5)] = p_temp;
				((voxel *)cptr)->update_self (mod.x/VOX_L, mod.y/VOX_L, mod.z/VOX_L);
				((voxel *)cptr)->update_vox (mod.x/VOX_L, mod.y/VOX_L, mod.z/VOX_L);
			}
	    }
	}
}

//! Appends points to the vector of points
/*! This method recursively calls voxel::all_points(), to append all the points in the leaf nodes to the vector.
*	This method is called from GPU_FE::Points()
*	Run by a single CUDA thread, since it is called only once and doesn't affect the performance.
*	\param vector of root voxels
*	\param size of the voxel
*	\param Tuple to store points
*	\see voxel::all_points(), GPU_FE::Points()
*/
__global__ void Print (Pair< long, Pair<voxel *, Point> > * v, long * s, Tuple * set) {
	int idx = 0;
	for (int i = 0; i < *s; i++) {
		float x = v[i].second.second.x;
		float y = v[i].second.second.y;
		float z = v[i].second.second.z;
		v[i].second.first->all_points(set, x, y, z, &idx);
	}
}

//! Wrapper class for occ_grid
/*! This class acts as an abstraction for the CUDA kernel methods. Also inherits virtual class Map_FE, so implements all its virtual methods.
*	\see Map_FE
*/
class GPU_FE : public Map_FE {

private:

	//thrust::device_vector< Pair< long, Pair<voxel *, Point> > > DV; // vector stored on device containing pairs of index and pointers to root voxels stored in device memory
	//! Vector in host memory containing root voxels.
	/*! The vector is sorted using the index of the root voxels and is copied on to a device-side vector before passing to the kernel methods.
	*/
	thrust::host_vector< Pair< long, Pair<voxel *, Point> > > HV; // vector stored on host containing pairs of index and pointers to root voxels stored in device memory
	//! Size of HV vector
	long s; // size of device_vector
	//! Temporary array stored in device memory.
	/*! This temporary array is used to store pointers to voxels created during current update on the device.
	*	\see Update_root
	*/
	Pair< long, Pair<voxel *, Point> > * dtemp; // temporary array to store pairs in kernel
	//! Temporary array stored in host memory.
	/*! This temporary array is used to copy the contents of dtemp vector and append them to HV vector.
	*/
	Pair< long, Pair<voxel *, Point> > * htemp; // temporary array to store pairs on host
	//! Pointer to depth image stored on device.
	unsigned short * D; // pointer to depth image stored in device
	//! Pointer to Pose struct stored on device.
	Pose * P; // pointer to pose stored in device
	//! Pointer ot Cam struct stored on device.
	Cam * C; // pointer to camera properties stored in device
	//! Size of HV vector; passed to device.
	long * S;

public:

	//! Default Constructor
	/*!	Static memory required for the device members are allocated on device memory.
	*	Space for temporary array on host is allocated in host heap memory.
	*/
	GPU_FE () {
		cudaMalloc ((void **) &D, w*h*sizeof(unsigned short));
		cudaMalloc ((void **) &P, sizeof(Pose));
		cudaMalloc ((void **) &C, sizeof(Cam));
		cudaMalloc ((void **) &dtemp, 125*sizeof(Pair< long, Pair<voxel *, Point> >));
		htemp = (Pair< long, Pair<voxel *, Point> > *) malloc(125*sizeof(Pair< long, Pair<voxel *, Point> >));
		cudaMalloc ((void **) &S, sizeof(long));
		s = 0l;

	}

	//! Updates the measurement data in the global map
	/*! Calls the global kernel method Update_root().
	*	Structs to be passed to the kernel are set up and the input parameters ae copied on to the device memory.
	*	After the call to the kernel has finished, the new root voxels are stored in HV and sorted by their indices.
	*	\param Camera object
	*	\param pose of T265
	*	\param 16-bit D435 depth image
	*	\see Update_root(), Map_FE::Update()
	*/
	void Update (Camera const &C, rs2_pose const &pose, cv::Mat const &depth) {
		quaternion q_T265 (pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w);
    	quaternion t_T265 (pose.translation.x, pose.translation.y, pose.translation.z, 0);
    	quaternion q_G_D435 = q_T265 * Q_T265_D435 * quaternion(1,0,0,0);
    	quaternion t_G_D435 = t_T265 + q_T265 * T_T265_D435 * q_T265.inv();

    	struct Cam c = {0, 0, 0, 0, 0, quaternion(0,0,0,0), quaternion(0,0,0,0)};
    	c.fx = C.fx; c.fy = C.fy;
    	c.ppx = C.ppx; c.ppy = C.ppy;
    	c.scale = C.scale;
    	c.Q_TD = Q_T265_D435; c.T_TD = T_T265_D435;

    	struct Pose p = {quaternion(0,0,0,0), quaternion(0,0,0,0)};
    	p.t = t_G_D435;
    	p.r = q_G_D435;

    	thrust::device_vector< Pair< long, Pair<voxel *, Point> > > DV(HV.begin(), HV.end());

		cudaMemcpy (this->C, &c, sizeof(Cam), cudaMemcpyHostToDevice);
		cudaMemcpy (this->P, &p, sizeof(Pose), cudaMemcpyHostToDevice);
		cudaMemcpy (this->D, depth.ptr<unsigned short>(0), w*h*sizeof(unsigned short), cudaMemcpyHostToDevice);
		Pair< long, Pair<voxel *, Point> > p_temp = Pair< long, Pair<voxel *, Point> >(0l, Pair<voxel *, Point>(NULL, Point{0,0,0}));
		for (int i = 0; i < 125; i++)
			htemp[i] = p_temp;
		cudaMemcpy (this->dtemp, htemp, 125*sizeof(Pair< long, Pair<voxel *, Point> >), cudaMemcpyHostToDevice);
		cudaMemcpy (this->S, &s, sizeof(long), cudaMemcpyHostToDevice);

		Update_root<<<NUM_B, NUM_T>>>(D, thrust::raw_pointer_cast(&DV[0]), S, dtemp, this->C, P);
		gpuCheckKernelExecutionError( __FILE__, __LINE__);

		cudaMemcpy (htemp, this->dtemp, 125*sizeof(Pair< long, Pair<voxel *, Point> >), cudaMemcpyDeviceToHost);
		for (int i = 0; i < 125; i++) {
			if ((void *)htemp[i].second.first != NULL) {
				HV.push_back (htemp[i]);
				s++;
			}
		}
		thrust::stable_sort (thrust::host, HV.begin(), HV.end());

	}

	//! Appends all points in global map to the vector.
	/*! This is a single threaded kernel method call.
	*	\param vector of points
	*	\see Print(), Map_FE::Points()
	*/
	void Points (std::vector < std::tuple<float, float, float, float> > * points) { // keep single threaded preferably
		Tuple set[100000];
		Tuple * cset;
		cudaMalloc ((void **) &cset, 100000*sizeof(Tuple));
		cudaMemcpy (S, &s, sizeof(long), cudaMemcpyHostToDevice);
		thrust::device_vector< Pair< long, Pair<voxel *, Point> > > DV(HV.begin(), HV.end());
		Print<<<1, 1>>> (thrust::raw_pointer_cast(&DV[0]), S, cset);
		cudaMemcpy (set, cset, 100000*sizeof(Tuple), cudaMemcpyDeviceToHost);
		int i = 0;
		while(i < 100000) {
			Tuple pt = set[i];
			if (pt.x != 0 || pt.y != 0 || pt.z != 0) {
				i++;
				points->push_back(std::make_tuple(pt.x, pt.y, pt.z, pt.c));
				continue;
			}
			break;
		}

	}

	//! Destructor
	/*! Deletes the global map
	*	\see Delete()
	*/
	~GPU_FE () { // keep single threaded preferably
		cudaFree(D);
		cudaFree(P);
		cudaFree(S);
		cudaFree(C);
		cudaFree(dtemp);
	}

};


#endif