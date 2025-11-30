#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

//* In report Explain what bitonic sort is,
//* explain our solution, put the code, then trace it

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

void randomNumbers(int* arr, int size){
    srand(time(NULL));    
    for(int i = 0; i < size ; i++){
        arr[i] = rand();
    }
}

void printArray(int *arr, int length){ // slow for large n
    printf("\n-------------------------------\n");
    for(int i=0; i < length; i++){

        printf("%d[%d]  ",i,arr[i]);
    }
}

void checkAscending(int* arr, int size, bool sortAscending){
    for(int i = 0; i < size-1; i++){
        if(sortAscending){
            if (arr[i] > arr[i+1])
            {
                printf("\n\nThere is a problem with sorting %d\n\n",i);
                return;
            }

        }else{
            if (arr[i] < arr[i+1])
            {
                printf("\n\nThere is a problem with sorting %d\n\n",i);
                return;
            }
        }
        
    }printf("\n\nAll Sorted!.\n\n");
}


__global__ void bitonic(int *data,
                        int step_i,
                        int seq_ij,
                        int active_range,
                        int length,
                        bool sortAscending
                    ) {
    // TODO:
    // 1. calculate Tk
    int Tk = blockIdx.x * blockDim.x + threadIdx.x;
    // 2. check i within bounds (i < N)
    if(Tk < length){
        // 4. decide if this thread should act (partner > i)
        if((Tk % seq_ij) < active_range){
            
            int p = Tk + active_range;  // partner

            // 6. compare data[i] and data[partner]
            int group = Tk >> step_i; // k / 2**i
            bool ascending = ((group % 2) == 0);
            if (sortAscending){
                // 7. swap if needed
                if(ascending){ // ascending
                    if(data[Tk] > data[p]){
                        int temp = data[Tk];
                        data[Tk] = data[p];
                        data[p] = temp;
                    }
                }else {     //descending
                    if(data[Tk] < data[p]){
                        int temp = data[Tk];
                        data[Tk] = data[p];
                        data[p] = temp;
                    }
                }
            }else{
                if(ascending){ // descending
                    if(data[Tk] < data[p]){
                        int temp = data[Tk];
                        data[Tk] = data[p];
                        data[p] = temp;
                    }
                }else {     //ascending
                    if(data[Tk] > data[p]){
                        int temp = data[Tk];
                        data[Tk] = data[p];
                        data[p] = temp;
                    }
                }

            }

        }
    }
}

int main() {

    const int LOGN = 3;
    const int length = 1 << LOGN; // 2**LOGN --> n

    const bool sortAscending = 0; // change to 0 for descending

    int h_data[length];
    printf("Length = %d\n", length);

    randomNumbers(h_data, length);
    printArray(h_data, length);

    int *d_data;
    int size = length * sizeof(int);

    int threadsPerBlock = 1024; 
    int blocks = (length + threadsPerBlock -1) / threadsPerBlock; //! length must 2**n

    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data,size, cudaMemcpyHostToDevice);
    


    //major step <<Step>>
    for(int i = 1; i <= LOGN; i++) { 
        

        //minor step <<Stage>>
        for(int j = 1; j <= i; j++){
            
            int seq_ij = 1 << (i - j + 1);     // 2 ** (i-j+1)
            int active_range = seq_ij >> 1;    // seq / 2

            bitonic<<<blocks, threadsPerBlock>>>(
                d_data,
                i,              //step_i
                seq_ij,         
                active_range,   
                length,          // n
                sortAscending
            );
            cudaDeviceSynchronize();
        }

    }
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);


    printArray(h_data, length);
    checkAscending(h_data, length, sortAscending);
    


    return 0;
}
