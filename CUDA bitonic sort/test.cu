#include <stdio.h>

//! NUM of element should be 2**n
//! for n elements we need log(n) steps
//! for step i there are i stage 
//! seq = 2**(i-j+1),  active_range = seq // 2

//! increase or decrease 
// if(k/2**i) is even tk will apply +
// if(k/2**i) is odd tk will apply -

//? witch data to use?
// if thread Tk is involved, it will process data[k] and data[k + (seqij /2)]

__global__ void helloCUDA() {
    
}



int main() {
    
    int size = 8 * sizeof(int);
    return 0;
}
