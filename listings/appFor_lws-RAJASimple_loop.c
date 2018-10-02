#include "vSched.h"
#define FORALL_BEGIN(strat, s,e, start, end, tid, numThds )  loop_start_ ## strat (s,e ,&start, &end, tid, numThds);  do {
#define FORALL_END(strat, start, end, tid)  } while( loop_next_ ## strat (&start, &end, tid));
void* dotProdFunc(void* arg)
{
int startInd =  (probSize*threadNum)/numThreads;
int endInd = (probSize*(threadNum+1))/numThreads;
 while(iter < numIters) {
    mySum = 0.0; //reset sum to zero at the beginning of the product
    if(threadNum == 0) sum = 0.0;
    if(threadNum == 0) setCDY(static_fraction, constraint, chunk_size);
#pragma omp parallel
     FORALL_BEGIN(statdynstaggered, 0, probSize, startInd, endInd, threadNum, numThreads)
     for (i = startInd ; i < endInd; i++)
	mySum += a[i]*b[i];
     FORALL_END(statdynstaggered, startInd, endInd, threadNum)
    pthread_mutex_lock(&myLock);
    sum += mySum;
    pthread_mutex_unlock(&myLock);
    pthread_barrier_wait(&myBarrier);
    if(threadNum == 0) iter++;
    pthread_barrier_wait(&myBarrier);
  } // end timestep loop
}

