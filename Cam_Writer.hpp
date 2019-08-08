#include <iostream> 
#include <sys/ipc.h> // ftok()
#include <sys/types.h> // ftok(),
#include <sys/mman.h> // shm_open()
#include <sys/stat.h> // shm_opem(), sem_open()
#include <fcntl.h> // shm_open(), sem_open()
#include <semaphore.h> // sem_open()
#include <string.h> // memcpy()
#include <stdio.h> 
#include <unistd.h> // close()

#include <opencv2/opencv.hpp>

#include <librealsense2/rs.hpp>     // Include RealSense Cross Platform API

#include "Camera.hpp"




class Writer {

public:

	rs2::frameset fs_d435, fs_t265;
	Camera C;
	Bool_Init b;

	int fd_d435, fd_t265; // file descriptors for shared memory
	void * ptr_d435, ptr_t265; // shared memory pointers
	sem_t * semptr_d435, semptr_t265; // semaphores for locking


	bool Init () {

		b = C.Init();

		fd_d435 = shm_open ("/D435", O_RDWR | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
		fd_t265 = shm_open ("/T265", O_RDWR | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
		if (fd_d435 < 0 || fd_t265 < 0)
        	return false;

		ptr_d435 = mmap (NULL, sizeof(rs2::frameset), PROT_READ | PROT_WRITE, MAP_SHARED, fd_d435, 0);
		ptr_t265 = mmap (NULL, sizeof(rs2::frameset), PROT_READ | PROT_WRITE, MAP_SHARED, fd_t265, 0);
		if (ptr_d435 == NULL || ptr_t265 == NULL)
        	return false;
        ftruncate (fd_d435, sizeof(rs2::frameset));
        ftruncate (fd_t265, sizeof(rs2::frameset));

        sem_t * semptr_d435 = sem_open ("/D435", O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH, 1);
        sem_t * semptr_t265 = sem_open ("/T265", O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH, 1);
        if (semptr_d435 == (void *) -1 || semptr_t265 == (void *) -1)
        	return false;

        return true;
	}
	
};