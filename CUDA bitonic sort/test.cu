#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024
//! NUM of element should be 2**n
//! for n elements we need log(n) steps
//! for step i there are i stage 
//! seq = 2**(i-j+1),  active_range = seq // 2

//? increase or decrease?
// if(k/2**i) is even tk will apply +
// if(k/2**i) is odd tk will apply -

//? witch data to use?
// if thread Tk is involved, it will process data[k] and data[k + (seqij /2)]

__global__ void bitonic(int *data, int j, int k, int length) {
    // TODO:
    // 1. calculate i (global thread index)
    int Tk = blockIdx.x * blockDim.x + threadIdx.x;
    // 2. check i within bounds (i < N)
    if(Tk < length){
        //2**(i-j+1)
        // 3. compute partner = i ^ j
        // ? int p = 1 << (Tk - j + 1);
        int p = Tk ^ j; // ^ XOR
        // 4. decide if this thread should act (partner > i)
        if(p > Tk){
            // 5. decide direction (ascending or descending) based on i & k
            // 6. compare data[i] and data[partner]
            // 7. swap if needed
            //! did't work k/ (1 << Tk)
            if( (Tk & k) == 0){ // ascending
                if(data[Tk] > data[p]){
                    int temp = data[Tk];
                    data[Tk] = data[p];
                    data[p] = temp;
                }
            }else { //descending
                if(data[Tk] < data[p]){
                    int temp = data[Tk];
                    data[Tk] = data[p];
                    data[p] = temp;
                }
            }

        }
    }
    // copy back result --> in the main
}

void randomNumbers(int* arr, int size){
    srand(time(NULL));    
    for(int i = 0; i < size ; i++){
        arr[i] = rand();
    }
}

void printArray(int *arr, int length){
    printf("\n-------------------------------\n");
    for(int i=0; i < length; i++){

        printf("%d[%d]  ",i,arr[i]);
    }
}

void checkAscending(int* arr, int size){
    for(int i = 0; i < size-1; i++){
        if (arr[i] > arr[i+1])
        {
            printf("\n\nThere is a problem with sorting %d\n\n",i);
            return;
        }
        
    }printf("\n\nAll Sorted!.\n\n");
}
int main() {
    const int LOGN = 18;
    const int rr = 1 << LOGN; // 2**LOGN
    int h_data[rr];
    int length = sizeof(h_data) / sizeof(h_data[0]);
    printf("length = %d\n", length);
    randomNumbers(h_data, length);
    printArray(h_data, length);
    int *d_data;
    int size = length * sizeof(int);
    int threadsPerBlock = 1024; //256
    int blocks = (length + threadsPerBlock -1) / threadsPerBlock; //! length must 2**n
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data,size, cudaMemcpyHostToDevice);
    // k = current subsequence length that we are working on (2,4,16,N)
    // j = current distance between elements that being compared
    //major step <<Step>>
    for(int k = 2; k <= length; k *= 2) { //  k <<=1 --> k = k << 1 shift to left
        //minor step <<Stage>>
        for(int j = k / 2; j > 0; j /= 2){
            bitonic<<<blocks, threadsPerBlock>>>(d_data,j,k,length);
            cudaDeviceSynchronize();
        }
        
    }
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    printArray(h_data, length);
    checkAscending(h_data, length);
    //cudaMalloc
    //cudaMemcpy
    //lunch kernel
    //cudaMemcpy
    //cudaFree
    return 0;
}
