__kernel void vector_add(__global const int *A, __global const int *B, __global int *C) {

    // Get the global index
    int i = get_global_id(0);

    // Do the operation
    C[i] = A[i] + B[i];
}
