void* dotProdFunc(void* arg)
{
  int startInd =  (probSize*threadNum)/numThreads;
  int endInd = (probSize*(threadNum+1))/numThreads;
  mySum = 0.0; //reset sum to zero at the beginning of the product
  if(threadNum == 0) sum = 0.0;
  pthread_barrier_wait(&myBarrier);
  FORALL_BEGIN(statdynstaggered, 0, probSize, startInd, endInd, threadNum, numThreads)
    for (i = startInd ; i < endInd; i++) mySum += a[i]*b[i];
  FORALL_END(statdynstaggered, startInd, endInd, threadNum)
  pthread_mutex_lock(&myLock);
  sum += mySum;
  pthread_mutex_unlock(&myLock);
  pthread_barrier_wait(&myBarrier);
  if(threadNum == 0)
    iter++;
  pthread_barrier_wait(&myBarrier);
}

