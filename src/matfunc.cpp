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

int solvetri(VectorXd*, VectorXd*);

int polynomialNeumann(VectorXd*);

int polynomialHorner(VectorXd*);

int main(int argc, char *argv[])
{
	//Defining variables to keep track of times
	clock_t time_init, time_end;
	double time_neumann=0.0, time_horner=0.0;

	//Defining the number of threads
	//omp_set_num_threads(5);
	//setNbThreads(5);
	int nthreads = nbThreads();
	cout << "The number of threads is " << nthreads << "." << endl;

	//Initiallizing the input and output matrices
	unsigned int msize = 100;
	inputmatrix = new MatrixXd(msize, msize);
	*inputmatrix = MatrixXd::Random(msize, msize);//MatrixXd::Random(msize, msize);
	//Setting up output matrix to store the final result
	outputmatrix = new MatrixXd(msize, msize);
	*outputmatrix = MatrixXd::Zero(msize, msize);

	//Calling the matrix polynomial evaluation function
	VectorXd coeffs(9);
	for(unsigned int i = 0; i < 9; i++) coeffs(i) = 2*i+1;
	time_init = clock();
	int presult = polynomialNeumann(&coeffs);
	time_end = clock();
	time_neumann = 1000*(time_end-time_init)/(double)CLOCKS_PER_SEC;
	if(presult) {
		cout << "The function plynomialNeumann return with code " << presult << "." << endl;
		exit(FAIL);
	}

	cout << "The function polynomialNeumann(...) had been completed." << endl;
	//Saving the polynomialNeumann matrix result
	MatrixXd neumannOutput = *outputmatrix;

	//Executing the polynomialHorner(...) function
	time_init = clock();
	presult = polynomialHorner(&coeffs);
	time_end = clock();
	time_horner = 1000*(time_end-time_init)/(double)CLOCKS_PER_SEC;
	//Saving the polynomialHorner matrix result
	MatrixXd hornerOutput = *outputmatrix;



	/*cout << "Input :" << endl;
	cout << *inputmatrix << endl;
	cout << "Horner:" << endl;
	cout << hornerOutput << endl;
	cout << "Neumann:" << endl;
	cout << neumannOutput << endl;*/
	cout << "The norm of the difference is:" << (hornerOutput-neumannOutput).norm() << "." << endl;
	cout << "Neumann time = " << time_neumann << " Horner time = " << time_horner << endl;

   /* Last thing that main() should do */
   pthread_exit(NULL);
}

void *neumann(void* data){
	/*
	 * Input:
	 * although we represent is as a void pointer, the inpurt argument is a pointer for a struct of type Stt.
	 * The element N represents the series size and the element alpha represents the constant to be mutiplied by the
	 * resulting Neumann series.
	 * Output:
	 * no output, but it modifies the matrix pointed by the global variable outputmatrix. The access synchronization is
	 * accomplished by means of mutual exclusion (mutex) variable.
	 * Description:
	 * This function computes the Neumann series of the input matrix efficiently
	 * up to size 9.
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
		break;
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

	//Scaling the tempm matrix before updating the outputmatrix
	tempm = alpha*tempm;

	//Synchronizing to update the matrix
	pthread_mutex_lock(&mutex);
	*outputmatrix += tempm;
	pthread_mutex_unlock(&mutex);

	//The last thing that the thread executes
	*status = SUCCESS;
	pthread_exit((void*) status);
}

int solvetri(VectorXd* a, VectorXd* sol){
	/*
	 * Input:
	 * the vector pointed by a represents the right-hand-side of the linear system associated with the matrix function
	 * computation through several Neumann series.
	 * Output:
	 * the vector pointed by sol is the solution of the linear system associated with the matrix function
	 * computation through several Neumann series.
	 * Description:
	 * this function solves the linear system associated with the matrix function computation through several Neumann series.
	 * The memmory for the solution vector is allocated by the calling function and must be deallocated by it too.
	 * The system is assumed to have a upper triangular matrix.
	 */

	if(sol->rows() != a->rows()) {
		cout << "Error: input vectors with different sizes" << endl;
		return FAIL;
	}

	//Using the analytical solution as described in the paper, we simply have
	for(unsigned int i = 0; i < a->rows()-1; i++){
		(*sol)(i) = (*a)(i)-(*a)(i+1);
	}
	(*sol)(a->rows()-1) = (*a)(a->rows()-1);

	return SUCCESS;
}

int polynomialNeumann(VectorXd* coeffs){
	/*
	 * Input:
	 * coeffs is a pointer for the vector representing the coefficients of the matrix polynomial.
	 * Its size is the polynomial size.
	 * Output:
	 * Description:
	 * this fucntion computes the matrix polynomial whose coefficients are given by the vector pointed by coeffs.
	 * The input matrix is the one pointed by the global variable inputmatrix and the output occurs in the one pointed
	 * by outputmatrix. All the computation is done using parallel computation which is accomplished through the implementation
	 * of the function neumann(...).
	 */

	//Sanity Check
	if(coeffs->rows() > 9) {
		cout << "Error: the matrix polynomial must have size no greater than 9." << endl;
		return FAIL;
	}

	//Defining the vector of thread handles
	pthread_t* threads = new pthread_t[coeffs->rows()];
	//Defining the threads attributes
	pthread_attr_t pattr;

	//Array of structures to be sent to the function neumann(...)
	Stt* datarray = new Stt[coeffs->rows()];

	//Solving he associated linear system
	VectorXd sol = VectorXd(coeffs->rows());
	int rsolve = solvetri(coeffs, &sol);

	if(rsolve == FAIL){
		cout << "Error: error during the solution of the linear system." << endl;
		exit(FAIL);
	}

	//Auxiliary variable storing the pthread_create return
	int rcreate;
	//Auxiliary variable storing thread status after execution
	void *thread_status = NULL;

	//Initializing thread attribute and setting it to be joinable
	pthread_attr_init(&pattr);
	pthread_attr_setdetachstate(&pattr, PTHREAD_CREATE_JOINABLE);
	//Initiatling the mutex
	pthread_mutex_init(&mutex, NULL);

	//Before the computation starts, the outputmatrx must be cleared
	*outputmatrix = MatrixXd::Zero(inputmatrix->rows(), inputmatrix->cols());

	for(unsigned int i = 0; i < coeffs->rows(); i++){
		//Building the data structure to be sent to the thread i
		Stt* data = &datarray[i];
		data->N = i+1;
		data->alpha = sol(i);
		cout << "In main: creating thread " << i << endl;
		rcreate = pthread_create(&threads[i], &pattr, neumann, (void *)data);
		if (rcreate){
			cout << "ERROR; return code from pthread_create() is " << rcreate << endl;
			exit(FAIL);
		}
	}

	cout << "All the threads are created" << endl;

	for(unsigned int i = 0; i < coeffs->rows(); i++){
		rcreate = pthread_join(threads[i], &thread_status);
		if(rcreate){
			cout << "Error: unsuccessful exectution of thread " << i << endl;
			exit(FAIL);
		}
		cout << "Thread " << i << " finished with status " << *((int*) thread_status) << endl;
	}

	//Freeing the thread attribute and mutex
	pthread_attr_destroy(&pattr);
	pthread_mutex_destroy(&mutex);
	//Deleting the created arrays
	delete [] threads;
	delete [] datarray;

	return SUCCESS;
}

int polynomialHorner(VectorXd* coeffs){
	/*
	 * Input:
	 * coeffs is a pointer for the vector representing the coefficients of the matrix polynomial.
	 * Its size is the polynomial size.
	 * Output:
	 * Description:
	 * this fucntion computes the matrix polynomial whose coefficients are given by the vector pointed by coeffs using Horner rule.
	 * The input matrix is the one pointed by the global variable inputmatrix and the output occurs in the one pointed
	 * by outputmatrix. Different from the other polynomialNeumann(...) function, this method can compute matrix polynomials of arbitrary size.
	 */

	//Defining temporary matrix
	MatrixXd tempm = MatrixXd(inputmatrix->rows(), inputmatrix->cols());
	tempm = MatrixXd::Zero(inputmatrix->rows(), inputmatrix->cols());

	unsigned int poln = coeffs->rows();

	//First Horner rule iteration
	tempm = (*coeffs)(poln-1)*(MatrixXd::Identity(inputmatrix->rows(), inputmatrix->cols()));
	//Remaining iterations
	for(int i = poln-2; i >= 0; i--){
		tempm = (*coeffs)(i)*(MatrixXd::Identity(inputmatrix->rows(), inputmatrix->cols()))+tempm*(*inputmatrix);
	}

	//Updating the outputmatrix
	*outputmatrix = tempm;

	return SUCCESS;
}
