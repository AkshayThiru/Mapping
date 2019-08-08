#ifndef VOXEL_H
#define VOXEL_H


#include <iostream>
#include <fstream>

#include <typeinfo>
#include <cmath>
#include <math.h>
#include <vector>
#include <array>
#include <tuple>
#include <algorithm>
#include <map>
#include <utility>
#include <unistd.h>

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
#define VOX_L 2.56 // edge length of root voxel in meter. Define as float:eg: 2.00
//! Variance of measurement
/*! This is the 3-D variance of each point measured. Assumed constant and isotropic.
*	The co-variance matrix in this case is \f$ VAR\_P . \mathbb{1}_{3{\times}3} \f$
*/
#define VAR_P 0.005 // variance of each measurement


//! A basic Quaternion class
/*! CPU: <br>
*	Quaternion class with components \f$ x, y, z, w \f$ such that \f$ q = x\textit{i} + y\textit{j} + z\textit{k} + w\f$.
*	Basic operators provided are \f$\times\f$: multiplication and \f$+\f$: addition. \f$^{-1}\f$: inverse is provided through quaternion::inv() method.
*/
class quaternion {

public:

	//! \name Components of quaternion.
	///@{
	float x, y, z, w;
	///@}

	//! Constructor taking x, y, z, w in order
	/*! Note: This is the only constructor provided.
	*	\param Components: \f$\textit{i}\f$, \f$\textit{j}\f$, \f$\textit{k}\f$, and \f$\mathbb{R}\f$
	*/
	quaternion (float x, float y, float z, float w) {
		this->x = x;
		this->y = y;
		this->z = z;
		this->w = w;
	}

	//! Inverse of the quaternion
	/*! To be used as q.inv() \f$\equiv q^{-1}\f$ 
	*	\return quaternion
	*/
	quaternion inv () {
		float l = (this->x)*(this->x) + (this->y)*(this->y) + (this->z)*(this->z) + (this->w)*(this->w);
		return quaternion (-(this->x)/l, -(this->y)/l, -(this->z)/l, (this->w)/l);
	}

	//! \f$\times\f$ operator
	/*	To be used as \f$q_1*q_2\f$ \f$\equiv q_1\timesq_2\f$, where &\f$q_1\f$ = this
	*	\param \f$q_2\f$
	*	\return quaternion
	*/
	quaternion operator * (quaternion const &q) {
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
	quaternion operator + (quaternion const &q) {
		return quaternion (this->x+q.x, this->y+q.y, this->z+q.z, this->w+q.w);
	}
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



/* leaf class */
//! Leaf nodes of the Octree structure
/*! CPU: <br>
*	This is not implemented as a voxel object because there can be millions of nodes and so the size should be as small as possible.
*	Stores the x, y, z co-ordinates of a single point inside it relative to edge length ie. \f$x, y, z \in [0,1)\f$. This is to maintain uniform accuracy
*	across all points. (accuracy of float type reduces as one moves away from 0)
*	The origin of the node is the vertex with all co-ordinates minimum. ie. if the origin of voxel is
*	\f$(x_o, y_o, z_o)\f$ and edge length is \f$L\f$, The vertices of the node are \f$\{(x_o, y_o, z_o), ..., (x_o+L, y_o+L, z_o+L)\}\f$
*	If the member leaf::_v \f$> 0\f$, the leaf node is occupied. If leaf::_v \f$= 0\f$, the leaf node is empty (this is not the same as unobserved. This means that
*	this node has been observed, but there is no point inside it). This has been used becuase if initially a node was observed to be empty, and containing a point afterwards, the same update
*	rule can be used without any change, in a single atomic operation. Although this is not particularly important for the CPU operation, it is extremely essential for the 
*	GPU operation to maintain consistency.
*	\see Voxel.cuh
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
	*	If the parameters provided are \f$(-1, -1, -,1)\f$, the node is set to be empty. Note that x_v, y_v, and z_v are set \f$= 0\f$.
	*	\param (x, y, z) relative to leaf node, ie. \f$x, y, z \in [0,1)\f$ for correct operation
	*/
	leaf (float x, float y, float z) { // state = -1: unoccupied
		_v = 0;
		x_v = y_v = z_v = 0;
		if (x != -1 && y != -1 && z != -1)
			this->update_leaf (x, y, z);
	}

	//! Update method for this node object.
	/*! Since every node contains only a single point, this update rule is used to combine the points into a single point.
	*	This is the same as the Measurement Update Step in EKF and SLAM. In this particular case the rule is a simple weighted average.
	*	So, if the point already existing in the node has a very low variance, the updated point will be very close to the previous point.
	*	Even if an anisotropic gaussian probability distribution function is used, the updated point will always be a convex combination of two points.
	*	\param (x, y, z) relative to leaf node, ie. \f$x, y, z \in [0,1)\f$ for correct operation
	*/
	void update_leaf (float x, float y, float z) { // x, y, z: scaled wrt to this->size
		x_v += x/VAR_P;
		y_v += y/VAR_P;
		z_v += z/VAR_P;
		_v += 1/VAR_P;
	}

};


/* voxel class */
//! Voxel/Intermediate nodes of the Octree structure
/*! CPU: <br>
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
	*	\see Voxel.cuh
	*/
	float _v; // inverse of variance

	//! Co-ordinates of a single point inside voxel node divided by the variance.
	/*! \name Co-ordinates
	*	The co-ordinates are measured relative to voxel node edge length, ie. \f$x, y, z \in [0,1)\f$.
	*	Note that although x_v, y_v, and z_v can are unbounded, the values of x, y, and z are bounded since the update is a convex combination of two points inside the node.
	*	The co-ordinates are divided by the variance so that the update can be performed in a single atomic operation while running in GPU.
	*	\see Voxel.cuh
	*/
	///@{

	float x_v, y_v, z_v; // point co-ordinate wrt voxel (0-1) / variance
	///@}
	//! Edge length of voxel node (\f$\textit{m}\f$)
	float size; // edge length of voxel in meter
	/* voxel * p; // parent voxel - initialize in constructor if used */


	//! Constructor for voxel node
	/*! Note that this is the only constructor provided.
	*	If the parameters provided are \f$(-1, -1, -,1)\f$, the node is set to be empty. Note that x_v, y_v, and z_v are set \f$= 0\f$.
	*	\param (x, y, z) relative to node, ie. \f$x, y, z \in [0,1)\f$ for correct operation
	*	\param edge length of voxel (\f$\textit{m}\f$)
	*/
	voxel (float x, float y, float z, float size) { // state = -1: unoccupied
		_v = 0;
		x_v = y_v = z_v = 0;
		this->size = size;
		c[0] = c[1] = c[2] = c[3] = c[4] = c[5] = c[6] = c[7] = NULL;
		if (x != -1 && y != -1 && z != -1)
			this->update_vox (x, y, z);
	}

	//! Update method for this node object.
	/*! For each voxel, two update steps are performed: one for the child voxel/leaf the input point lies in, and one for this voxel object.
	*	For the child update, it is first checked whether the child exists. If it does, leaf::update_leaf() or voxel::update_vox() is called on the child object.
	*	If it doesn't, a new child voxel/leaf is created and the constructor leaf::leaf() or voxel::voxel() is called.
	*	This step is a recursive one.
	*	The decision of whether the child is a voxel node or a leaf node is made considering the edge lengths of the children. (\f$=\frac{this\to\_v}{2}\f$)
	*	If child edge length \f$ \leq \f$ MIN_L, the child is a leaf node, else it is a voxel node.
	*	The next step is self update which is similar to leaf::update_leaf()
	*	\param (x, y, z) relative to node, ie. \f$x, y, z \in [0,1)\f$ for correct operation
	*	\see leaf::update_leaf()
	*/
	void update_vox (float x, float y, float z) { // x, y, z: scaled wrt to this->size

		/* update child voxels */
		int idx = (z >= 0.5)<<2 | (y >= 0.5)<<1 | (x >= 0.5); // idx of child voxel the point lies in

		if (size/4 >= MIN_L) { /* child is a voxel object */
			if (c[idx] == NULL)
				c[idx] = (void *) new voxel (fmodf(x,0.5)*2, fmodf(y,0.5)*2, fmodf(z,0.5)*2, size/2);
			else
				((voxel *)c[idx])->update_vox (fmodf(x,0.5)*2, fmodf(y,0.5)*2, fmodf(z,0.5)*2);
		}
		else { /* child is a leaf object */
			if (c[idx] == NULL)
				c[idx] = (void *) new leaf (fmodf(x,0.5)*2, fmodf(y,0.5)*2, fmodf(z,0.5)*2);
			else
				((leaf *)c[idx])->update_leaf (fmodf(x,0.5)*2, fmodf(y,0.5)*2, fmodf(z,0.5)*2);
		}

		/* update self */
		x_v += x/VAR_P;
		y_v += y/VAR_P;
		z_v += z/VAR_P;
		_v += 1/VAR_P;

	}

	//! Recursively frees up memory inside this voxel node.
	/*! This is called upon by the member method occ_grid::free_mem() (which is inturn called by CPU_FE::~CPU_FE()) on each of the root voxel nodes, which recursively deletes all the nodes in the octree.
	*	\see occ_grid::free_mem(), CPU_FE::~CPU_FE()
	*/
	void free_mem () {
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
	/*	This is called by occ_grid::all_points() (inturn called by CPU_FE::Points(), which can be user called or called by Logger::Close()) on each
	*	root voxel node, which recursively appends all points to the vector set.
	*	\param co-ordinates of points
	*	\param origin of the voxel node.
	*	\see occ_grid::all_points(), CPU_FE::Points(), Logger::Close()
	*/
	void all_points (std::vector < std::tuple<float, float, float, float> > * set, float x_o, float y_o, float z_o) {
		if (size/4 >= MIN_L) { /* child is a voxel object */
			for (int i = 0; i < 8; i++) {
				if (c[i] != NULL) {
					((voxel *)c[i])->all_points(set, x_o+size/2*(i&1), y_o+size/2*((i&2)>>1), z_o+size/2*((i&4)>>2));
				}
			}
		}
		else { /* child is a leaf object */
			leaf * p = NULL;
			for (int i = 0; i < 8; i++) {
				if (c[i] != NULL) {
					p = (leaf *) c[i];
					set->push_back ( std::make_tuple (x_o+((p->x_v)/(p->_v)+(i&1))*size/2, y_o+((p->y_v)/(p->_v)+((i&2)>>1))*size/2, z_o+((p->z_v)/(p->_v)+((i&4)>>2))*size/2, 1/(p->_v)) );
				}
			}
		}
	}

	//! Checks if this node has been observed or not.
	/*! If the node has atleast one filled or empty children, this method returns false.
	*	\see voxel
	*/
	bool is_empty () {
		for (int i = 0; i < 8; i++) {
			if (c[i] != NULL)
				return false;
		}
		return true;
	}

};


/* occ_grid class */
//! The top-most class managing the global map
/*!	An object of this class maintains the map.
*	This class is specific to the CPU mode of operation and can be thought of as an interface between the user and the global map.
*	A map, which is a red-black tree, is maintained, containing all the root voxels in the map.
*	The equivalent of this class for GPU code are the __global__ methods called from the host on the device.
*/
class occ_grid {

public:

	//! Array of pointers and origins of root voxels.
	/*!	This map contains an index calculated from the origin of the root voxel as the key, and a pair containing pointer to root voxel
	*	and the co-ordinates of the origin of the voxel.
	*	A key-value paradigm is used in order to implement a red-black tree, which brings down look-up time from \f$O(n)\f$ to \f$O(\ln(n))\f$.
	*	The index is a unique whole number calculated using the origin of the voxel.
	*	\see occ_grid::index()
	*/
	std::map < unsigned long, std::pair<voxel *, std::array<float, 3>> > root;

	//! Default Constructor
	occ_grid () {
		root.clear();
	}

	//! Updates point in the global map.
	/*! This method recursively calls voxel::update_vox(), to update the point in the respective voxel. This method itself is called upon by CPU_FE::Update().
	*	The information on the origin of the voxel is used to identify the voxel, and the index is used as a key to search in the red-black tree.
	*	This method is the same as voxel::update_vox(), other than the fact that the point doesn't directly map to any "child" voxel.
	*	\param global co-ordinates of the point to be updated
	*	\see index(), voxel::update_vox(), CPU_FE::Update()
	*/
	void update_point (float x, float y, float z) { // x, y, z are in global co-ordinates
		std::array<float, 3> mod = this->mod_p(std::array<float, 3> {x, y, z});
		unsigned long idx = this->index(std::array<float, 3> {x, y, z});

		auto itr = root.find(idx);
		if (itr != root.end()) { /* root voxel containing point exists */
			itr->second.first->update_vox(mod[0]/VOX_L, mod[1]/VOX_L, mod[2]/VOX_L);
		}
		else { /* root voxel doesn't exist */
			voxel * r = new voxel (mod[0]/VOX_L, mod[1]/VOX_L, mod[2]/VOX_L, VOX_L);
			std::array<float, 3> l {x-mod[0], y-mod[1], z-mod[2]};
			root.insert( std::pair< unsigned long, std::pair<voxel *, std::array<float, 3>> >(idx, std::pair<voxel *, std::array<float, 3>>(r, l)) );
		}
	}

	//! Appends points to the vector of points
	/*! This method recursively calls voxel::all_points(), to append all the points in the leaf nodes to the vector.
	*	This method is called from CPU_FE::Points()
	*	\param vector of point co-ordinates
	*	\see voxel::all_points(), CPU_FE::Points()
	*/
	void all_points (std::vector < std::tuple<float, float, float, float> > * set) {
		std::map<unsigned long, std::pair<voxel *, std::array<float, 3>>>::iterator itr;
		for (itr = root.begin(); itr != root.end(); itr++){
			itr->second.first->all_points(set, itr->second.second[0], itr->second.second[1], itr->second.second[2]);
		}
	}

	//! Deletes the global map
	/*! This method recursively calls voxel::free_mem(), to delete all the nodes in the octree.
	*	This method is called from CPU_FE::~CPU_FE()
	*	\see voxel::free_mem(), CPU_FE::~CPU_FE()
	*/
	void free_mem () {
		std::map<unsigned long, std::pair<voxel *, std::array<float, 3>>>::iterator itr;
		for (itr = root.begin(); itr != root.end(); itr++){
			itr->second.first->free_mem();
		}
	}

	/*void seed_unoccupied (std::vector< std::array<float, 3> > P) { // vector of points: camera, co-ordinates of (0,0), (w,0), (w,h), (0,h) at max depth
		std::map< unsigned long, std::pair<voxel *, std::array<float, 3>> > * pre, * cur;
		auto itr = root.find(this->index(P[0]));
		cur->insert( std::pair< unsigned long, std::pair<voxel *, std::array<float, 3>> >(itr->first, itr->second) );
		this->fill_unocuupied (pre, cur, &P);
	}

	void fill_unocuupied (std::map< unsigned long, std::pair<voxel *, std::array<float, 3>> > * pre, std::map< unsigned long, std::pair<voxel *, std::array<float, 3>> > * cur, std::vector< std::array<float, 3> > * P) {
		;
	}*/

	//! Calculates index used as key to index into root.
	/*! This is used to calculate a unique whole number from a set of three integers: indices of origin of the voxel.
	*	Instead of using three nested maps each trying to index one co-ordinate at each level (\f$ O(\ln(N_x)+\ln(N_y)+\ln(N_z))\f$), 
	*	a bijective mapping from \f$ \mathbb{Z}^{3} \to \mathbb{N}\f$ is defined. Although the order of the complexity remains the same, the look-up is guaranteed to occur in less time than the previous case.
	*	\param co-ordinates of the origin of voxel
	*	\return index of point
	*	\see occ_grid::update_point(), root
	*/
	unsigned long index (std::array<float, 3> p) {
		std::array<float, 3> mod = this->mod_p(p);
		unsigned long a = (p[0] < 0) ? -2*std::round((p[0]-mod[0])/VOX_L)-1 : 2*std::round((p[0]-mod[0])/VOX_L);
		unsigned long b = (p[1] < 0) ? -2*std::round((p[1]-mod[1])/VOX_L)-1 : 2*std::round((p[1]-mod[1])/VOX_L);
		unsigned long c = (p[2] < 0) ? -2*std::round((p[2]-mod[2])/VOX_L)-1 : 2*std::round((p[2]-mod[2])/VOX_L);
		unsigned long idx = (a+b+c+2)*(a+b+c+1)*(a+b+c)/6 + (a+b+1)*(a+b)/2 + a;
		return idx;
	}

	//! Calculates co-ordinate of point modulo edge length.
	/*! Returns \f$p \mod VOX\_L[0, 1)^3\f$
	*	\param co-ordinate of point
	*	\return modulo of co-ordinate of point
	*/
	std::array<float, 3> mod_p (std::array<float, 3> p) {
		return std::array<float, 3> {fmodf(fmodf(p[0], VOX_L) + VOX_L, VOX_L), fmodf(fmodf(p[1], VOX_L) + VOX_L, VOX_L), fmodf(fmodf(p[2], VOX_L) + VOX_L, VOX_L)};
	}

};


//! Wrapper class for occ_grid
/*! This class acts as an abstraction for the occ_grid class. Also inherits virtual class Map_FE, so implements all its virtual methods.
*	\see Map_FE
*/
class CPU_FE : public Map_FE {

private:

	//! Global map object
	/*! \see occ_grid
	*/
	occ_grid * g_map;

public:

	//! Default Constructor
	CPU_FE () {
		g_map = new occ_grid();
	}

	//! Updates the measurement data in the global map
	/*! Sequencially calls occ_grid::update_point() on all points in the depth image.
	*	The co-ordinates are transformed from the D435 frame to T265 global frame and then passed on to occ_grid::update_point().
	*	\param Camera object
	*	\param pose of T265
	*	\param 16-bit D435 depth image
	*	\see occ_grid::update_point(), Map_FE::Update()
	*/
	void Update (Camera const &C, rs2_pose const &pose, cv::Mat const &depth) {
		quaternion q_T265 (pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w);
	    quaternion t_T265 (pose.translation.x, pose.translation.y, pose.translation.z, 0);
	    quaternion q_G_D435 = q_T265 * Q_T265_D435 * quaternion(1,0,0,0);
	    quaternion t_G_D435 = t_T265 + q_T265 * T_T265_D435 * q_T265.inv();
	    quaternion pose_pix (0, 0, 0, 0);

	    float x_D435, y_D435, z_D435;
	    for (int i = 0; i < h; i++) {
	        for (int j = 0; j < w; j++) {
	            z_D435 = depth.at<unsigned short int>(i,j) * C.scale;
	            x_D435 = (j-C.ppx)/C.fx * z_D435;
	            y_D435 = (i-C.ppy)/C.fy * z_D435;

	            pose_pix = t_G_D435 + q_G_D435 * quaternion(x_D435,y_D435,z_D435,0) * q_G_D435.inv();

	            if (z_D435 > D435_MIN && z_D435 < D435_MAX)
	                g_map->update_point (pose_pix.x, pose_pix.y, pose_pix.z);
	        }
	    }
	}

	//! Appends all points in global map to the vector.
	/*! \param vector of points
	*	\see occ_grid::all_points(), Map_FE::Points()
	*/
	void Points (std::vector < std::tuple<float, float, float, float> > * points) {
		g_map->all_points(points);
	}

	//! Destructor
	/*! Deletes the global map
	*	\see occ_grid::free_mem()
	*/
	~CPU_FE () {
		g_map->free_mem();
	}

};


#endif