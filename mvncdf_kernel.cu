//mvncdf_kernel.cu
//CUDA code of the Matlab GPU kernel to calculate the CDF of Gaussian using Monte Carlo.
//CUDA Montecarlo is similar to MapReduce. It can be called Transform-Combine.
//This kernel is called within Matlab and does the transform (Map) part. The rest can be done in Matlab.
//To be used as a Matlab CUDA kernel

__device__
size_t getGlobalIndex()
{
	// block number
	size_t const globalBlockIndex = blockIdx.x + blockIdx.y * gridDim.x;
	// thread number within the block
	size_t const localThreadIdx = threadIdx.x + blockDim.x * threadIdx.y;
	// The size of each block
	size_t const threadsPerBlock = blockDim.x*blockDim.y;
	// thread number (overall)
	return localThreadIdx + globalBlockIndex*threadsPerBlock;
}

// CDF
__device__
double normp(double z)
{
	return 0.5 * erfc(-z / sqrt((double)2));
}

__device__
double normq(double p)
{
	return -sqrt((double)2)*erfcinv(2*p);
}

// This function is repeated for each MC sample.
__device__
double transf
(
	const double * w,
	const double * b,
	const double * C ,
	unsigned int const m )
{
	// Initialise:
	double emd = normp(b[1-1]);
	double T = emd;
	#define MAXDIM 25 //Maximum dimensions. This can limits the available shared memory. 
	double y[MAXDIM]; 
	for(int j=0; j<m; j++)
		y[j]=0;
	unsigned int i = 2;
	while ( i <= m )
	{
		#define EPS_GPU 1e-7

		const double mx0 = emd * w[i-1-1];
		const double z = min(max(mx0, EPS_GPU), 1-EPS_GPU);
		y[i-1-1] = normq(z);
		double ysum=0;
		for( int j=0; j<m; j++)
		{
			ysum += y[j] * C[j+(i-1)*m];
		}
		emd=normp(b[i-1]-ysum);
		T *= emd;
		i++;
	}		
	return T;
}

//The current thread reads and writes on a speficic GPU (global) memoey location specified by the following function. Then it calls transf to do the actual work.
//This function is automatically called many times in parallel by all GPU kernels.
__global__
void processOneSample(
	double * out,
	const double *wa,
	const double *b,
	const double *C,
	const unsigned int m,
	const unsigned int numel
)
{
	// global thread number
	size_t const globalThreadIdx = getGlobalIndex();
	if (globalThreadIdx >= numel) {
		return;
	}

	//Locate the input in the relevant memory location
	double const *w = &( wa[globalThreadIdx*m] ); // m elements (in fact (m-1) is needed

	// Run the transformation (Map) on one (this) MC point
	double const T = transf( w, b, C, m );

	//Store the output
	out[globalThreadIdx] = T;

}
