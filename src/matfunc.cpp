/*
 * matfunc.cpp
 *
 *  Created on: May 27, 2017
 *      Author: Diego Coelho, PhD Candidate, UofC
 */

#include <iostream>
#include <ctime>
#include <string>
#include <pthread.h>
#include <cstdlib>
#include <cstdio>

#include "../../home/diego/softwares/eigen3.3/Eigen/Dense"

#define SUCCESS 0
#define FAIL -1
#define FILE_FAIL -2
#define REP 10000 //The number of replicates for simulation

#define NUM_THREADS	9

using namespace std;
using namespace Eigen;

//Defining global variables to be store the input and output matrix
MatrixXd* inputmatrix;
MatrixXd* outputmatrix;

//Defining mutex to control the update of the output matrix
pthread_mutex_t mutex;

//Defining the structure to be passed to the neumann(...) function
typedef struct {
	//Series size
	unsigned int N;
	//Real constant multiplying the Neumann series of the input matrix
	double alpha;
} Stt;

void *neumann(void* datatoprocess);

int main(int argc, char *argv[])
{
	//Defining the vector of thread handles
	pthread_t threads[NUM_THREADS];
	//Defining the threads attributes
	pthread_attr_t pattr;

	//Array of structures to be sent to the functio neumann(...)
	Stt datarray[NUM_THREADS];

	//Auxiliary variable storing the pthread_create return
	int rcreate;
	//Auxiliary variable storing thread status after execution
	void *thread_status = NULL;

	//Matrix size to be simulated
	unsigned int msize = 3;

	//Initializing thread attribute and setting it to be joinable
	pthread_attr_init(&pattr);
	pthread_attr_setdetachstate(&pattr, PTHREAD_CREATE_JOINABLE);
	//Initiatling the mutex
	pthread_mutex_init(&mutex, NULL);

	//Setting up the input matrix to be submitted to the Neumann series computation
	inputmatrix = new MatrixXd(msize, msize);
	*inputmatrix = MatrixXd::Random(msize, msize);
	//Setting up output matrix to store the final result
	outputmatrix = new MatrixXd(msize, msize);
	*outputmatrix = MatrixXd::Zero(msize, msize);

	for(unsigned int i = 0; i < NUM_THREADS; i++){
		//Building the data structure to be sent to the thread i
		Stt* data = &datarray[i];
		data->N = i+1;
		data->alpha = 1.0;
		cout << "In main: creating thread " << i << endl;
		rcreate = pthread_create(&threads[i], &pattr, neumann, (void *)data);
		if (rcreate){
			cout << "ERROR; return code from pthread_create() is " << rcreate << endl;
			exit(-1);
		}
	}

	cout << "All the threads are created" << endl;

	for(unsigned int i = 0; i < NUM_THREADS; i++){
		cout << "Checking the thread " << i << "." << endl;
		rcreate = pthread_join(threads[i], &thread_status);
		if(rcreate){
			cout << "Error: unsuccessful exectution of thread " << i << endl;
			exit(-1);
		}
		cout << "Thread " << i << " finished with status " << *((int*) thread_status) << endl;
	}

	//Freeing the thread attribute and mutex
	pthread_attr_destroy(&pattr);
	pthread_mutex_destroy(&mutex);

	cout << "Input and output matrices are:" << endl;
	cout << *inputmatrix << endl;
	cout << " and " << endl;
	cout << *outputmatrix << endl;

   /* Last thing that main() should do */
   pthread_exit(NULL);
}



void *neumann(void* data){
	/*TOUPDATE
	 * Input:
	 * inputmatrix is a pointer for a real square matrix of arbitrary size
	 * N is a unsigned int representing the series size
	 * Output:
	 * outputmatrix is a pointer for a real square matrix of arbitrary size
	 * Description:
	 * This function computes the Neumann series of the input matrix efficiently
	 * up to size 9. The memmory for the output matrix is
	 * allocated by the function and must be deallocated by the caling function.
	 */

	//Converting the input structure to NeumannData type
	Stt* mydata = (Stt*) data;
	//Passing the series size
	unsigned int N = mydata->N;
	//Passing the real constant multipliying the Neumann series of the input matrix
	double alpha = mydata->alpha;

	//Defining output status
	int* status = new int;

	//Sanity check
	if(inputmatrix->rows() == 0 || inputmatrix->cols() == 0){
		cout << "Error: input matrix can not be empty." << endl;
		*status = FAIL;
		pthread_exit((void*) status);
	} else if(inputmatrix->rows() != inputmatrix->cols()){
		cout << "Error: input matrix need to be square." << endl;
		*status = FAIL;
		pthread_exit((void*) status);
	} else if(N > 9){
		cout << "Error: the series size must be smaller than 10." << endl;
		*status = FAIL;
		pthread_exit((void*) status);
	}

	//Defining dentity matrix of the same size
	MatrixXd identity = MatrixXd::Identity(inputmatrix->rows(), inputmatrix->cols());

	//Defining auxiliary matrices
	MatrixXd squared(inputmatrix->rows(), inputmatrix->cols());
	MatrixXd cubed(inputmatrix->rows(), inputmatrix->cols());


	//Defining the temporary matrix that will be added to the global outputmatrix
	MatrixXd tempm = MatrixXd(inputmatrix->rows(), inputmatrix->cols());

	//Switch case for each differen series size
	switch(N){
	case (1):
		tempm = identity;
		break;
	case (2):
		tempm = identity+(*inputmatrix);
		break;
	case (3):
		tempm = identity+(*inputmatrix)+(*inputmatrix)*(*inputmatrix);
	case (4):
		squared = (*inputmatrix)*(*inputmatrix);
		tempm = (identity+(*inputmatrix))*(identity+squared);
		break;
	case (5):
		squared = (*inputmatrix)*(*inputmatrix);
		tempm = identity+(identity+squared)*((*inputmatrix)+squared);
		break;
	case (6):
		squared = (*inputmatrix)*(*inputmatrix);
		cubed = squared*(*inputmatrix);
		tempm = (identity+(*inputmatrix)+squared)*(identity+cubed);
		break;
	case (7):
		squared = (*inputmatrix)*(*inputmatrix);
		cubed = squared*(*inputmatrix);
		tempm = identity+((*inputmatrix)+squared+cubed)*(identity+cubed);
		break;
	case (8):
		squared = (*inputmatrix)*(*inputmatrix);
		tempm = (identity+(*inputmatrix))*(identity+squared)*(identity+squared*squared);
		break;
	case (9):
		squared = (*inputmatrix)*(*inputmatrix);
		cubed = squared*(*inputmatrix);
		tempm = (identity+(*inputmatrix)+squared);
		squared = cubed*cubed;
		tempm *= (identity+cubed+squared);
		break;
	default:
		cout << "Error: the size your selected does not match the options" << endl;
		*status = FAIL;
		pthread_exit((void*) status);
	}

	tempm = alpha*tempm;

	//Synchronizing to update the matrix
	pthread_mutex_lock(&mutex);
	*outputmatrix += tempm;
	pthread_mutex_unlock(&mutex);

	cout << "inside size " << N << endl;
	//The last thing that the thread executes
	*status = SUCCESS;
	pthread_exit((void*) status);
}
