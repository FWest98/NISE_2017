#include <stdlib.h>
#include <math.h>
#include "NISE_subs_gpu.cuh"
#include "util/cudaheader.cuh"

// Index triangular matrix
// Put in the .h file to allow external referencing
inline int Sindex(int a, int b, int N) { // inline to make it quicker
    int ind;
    if (a > b) {
        //ind=a+N*b-(b*(b+1)/2);
        ind = a + b * ((N << 1) - b - 1) / 2;
    }
    else {
        //ind=b+N*a-(a*(a+1)/2);
        ind = b + a * ((N << 1) - a - 1) / 2;
    }
    return ind;
}

/* Calculate occupancy */

/* Propagate doubles using diagonal vs. coupling sparce algorithm */
// Kernels

__global__ void expDiag(float *re, float *im, int n, int f, float *H0) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index > n) return;
    re[index] = cos(0.5f * H0[index] * f);
    im[index] = -sin(0.5f * H0[index] * f);
}

__global__ void multCVec(float *resR, float *resI, float *xR, float *xI, float *yR, float *yI, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index > n) return;
    resR[index] = xR[index] * yR[index] - xI[index] * yI[index];
    resI[index] = xR[index] * yR[index] + xI[index] * yI[index];
}

// Code
void propagate_vec_coupling_S_doubles_GPU(t_non* non, float* Hamiltonian_i, float* cr, float* ci, int m, float* Anh) {
    int N = non->singles;
    int N2 = N * (N + 1) / 2;
    const float f = non->deltat * icm2ifs * twoPi / m;
    float* H0 = (float *) calloc(N2, sizeof(float));
    float* H1 = (float *) calloc(N * N / 2, sizeof(float));
    int* col = (int *) calloc(N * N / 2, sizeof(int));
    int* row = (int *) calloc(N * N / 2, sizeof(int));
    float* re_U = (float *) calloc(N2, sizeof(float));
    float* im_U = (float *) calloc(N2, sizeof(float));
    float* ocr = (float *) calloc(N2, sizeof(float));
    float* oci = (float *) calloc(N2, sizeof(float));

    /* Build Hamiltonians H0 (diagonal) and H1 (coupling) */
    for (int a = 0; a < N; a++) {
        const int indexa = Sindex(a, a, N);
        for (int b = a; b < N; b++) {
            int index = Sindex(a, b, N);
            H0[index] = Hamiltonian_i[indexa] + Hamiltonian_i[Sindex(b, b, N)]; // Diagonal
            if (a == b) {
                if (non->anharmonicity == 0) {
                    H0[index] -= Anh[a];
                }
                else {
                    H0[index] -= non->anharmonicity;
                }
            }
        }
    }

    /* Build Hamiltonian H1 (coupling) */
    int kmax = 0;
    for (int a = 0; a < N; a++) {
        for (int b = a + 1; b < N; b++) {
            int index = b + a * ((N << 1) - a - 1) / 2; // Part of Sindex, but b > a is always true here

            if (fabsf(Hamiltonian_i[index]) > non->couplingcut) {
                H1[kmax] = Hamiltonian_i[index];
                col[kmax] = a, row[kmax] = b;
                kmax++;
            }
        }
    }

    // Move data to GPU
    float* re_U_G, * im_U_G, * cr_G, * ci_G, * ocr_G, * oci_G, * H0_G;
    cudaMalloc(&re_U_G, N2 * sizeof(float)); cudaMemcpy(re_U_G, re_U, N2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&im_U_G, N2 * sizeof(float)); cudaMemcpy(im_U_G, im_U, N2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&cr_G, N2 * sizeof(float)); cudaMemcpy(cr_G, cr, N2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&ci_G, N2 * sizeof(float)); cudaMemcpy(ci_G, ci, N2 * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMalloc(&ocr_G, N2 * sizeof(float)); cudaMemcpy(ocr_G, ocr, N2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&oci_G, N2 * sizeof(float)); cudaMemcpy(oci_G, oci, N2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&H0_G, N2 * sizeof(float)); cudaMemcpy(H0_G, H0, N2 * sizeof(float), cudaMemcpyHostToDevice);

    int gridSize = (N2 + 256 - 1) / 256;

    /* Exponentiate diagonal [U=exp(-i/2h H0 dt)] */
    expDiag KERNEL_ARG2(gridSize, 256) (re_U_G, im_U_G, N2, f, H0_G);
    /*for (int a = 0; a < N2; a++) {
        re_U[a] = cosf(0.5f * H0[a] * f);
        im_U[a] = -sinf(0.5f * H0[a] * f);
    }*/

    cudaMemcpy(re_U, re_U_G, N2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(im_U, im_U_G, N2 * sizeof(float), cudaMemcpyDeviceToHost);


    for (int i = 0; i < m; i++) {

        /* Multiply on vector first time */
        for (int a = 0; a < N2; a++) {
            ocr[a] = cr[a] * re_U[a] - ci[a] * im_U[a];
            oci[a] = cr[a] * im_U[a] + ci[a] * re_U[a];
        }

        /* Account for couplings */
        /* Loop over couplings */
        for (int k = 0; k < kmax; k++) {
            int a = col[k];
            int b = row[k];
            float J = H1[k] * f;

            /* Loop over wave functions <ca|Hab|cb> and <cb|Hba|ca> */
            // TODO speedup
            for (int c = 0; c < N; c++) {
                float si = (c == a || c == b) ? -sinf(J * sqrt2) : -sinf(J);

                float co = sqrtf(1 - si * si);
                int index1 = Sindex(a, c, N), index2 = Sindex(c, b, N);
                float cr1 = co * ocr[index1] - si * oci[index2];
                float ci1 = co * oci[index1] + si * ocr[index2];
                float cr2 = co * ocr[index2] - si * oci[index1];
                float ci2 = co * oci[index2] + si * ocr[index1];
                ocr[index1] = cr1, oci[index1] = ci1, ocr[index2] = cr2, oci[index2] = ci2;
            }
        }

        /* Multiply on vector second time */
        for (int a = 0; a < N2; a++) {
            cr[a] = ocr[a] * re_U[a] - oci[a] * im_U[a];
            ci[a] = ocr[a] * im_U[a] + oci[a] * re_U[a];
        }
    }
    free(ocr), free(oci), free(re_U), free(im_U), free(H1), free(H0);
    free(col), free(row);
}
