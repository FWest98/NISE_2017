#include "types.h"
#include <cstdlib>
#include <arrayfire.h>
#include "NISE_subs.h"

// Helper function to allocate a 2D array
template <class T> T** make2Darray(const int length) {
    auto out = new T*[length]();
    out[0] = new  T[length * length]();

    for (auto i = 1; i < length; i++) out[i] = out[i - 1] + length;

    return out;
}

// Helper function to free a 2D array
template <class T> void free2Darray(T** arr) {
    delete[] arr[0]; delete[] arr;
}

// Function to convert a symmetric matrix from 1D representation to more convenient 2D representation
template <class T> void convert1Dto2Dsymm(T* in, T** out, const int length) {
    for(auto i = 0; i < length; i++) {
        for (auto j = i; j < length; j++) {
            out[i][j] = in[j + i * ((length << 1) - i - 1) / 2];
        }
    }
}

// Function to convert a symmetric matrix from 2D to 1D rep
template <class T> void convert2Dto1Dsymm(T** in, T* out, const int length) {
    for(auto i = 0; i < length; i++) {
        for(auto j = i; j < length; j++) {
            out[j + i * ((length << 1) - i - 1) / 2] = in[i][j];
        }
    }
}

void propagate_vec_coupling_S_doubles_GPU(t_non* non, float* Hamiltonian_i_1D, float* cr_1D, float* ci_1D, int m, float* Anh) {
    auto N = non->singles;
    auto N2 = N * (N + 1) / 2;
    const auto f = non->deltat * icm2ifs * twoPi / m;
    auto H0 = make2Darray<float>(N);
    float* H1 = new float[N * N / 2]();
    int* col = new int[N * N / 2]();
    int* row = new int[N * N / 2]();

    auto c_1D = new af::cfloat[N2]();
    for(auto i = 0; i < N2; i++) {
        c_1D[i] = af::cfloat(cr_1D[i], ci_1D[i]);
    }

    auto c = make2Darray<af::cfloat>(N);
    convert1Dto2Dsymm(c_1D, c, N);

    auto Hamiltonian_i = make2Darray<float>(N);
    convert1Dto2Dsymm(Hamiltonian_i_1D, Hamiltonian_i, N);

    /* Build Hamiltonians H0 (diagonal) and H1 (coupling) */
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            H0[i][j] = Hamiltonian_i[i][i] + Hamiltonian_i[j][j]; // Diagonal
            if (i == j) {
                if (non->anharmonicity == 0) {
                    H0[i][j] -= Anh[i];
                }
                else {
                    H0[i][j] -= non->anharmonicity;
                }
            }
        }
    }

    /* Build Hamiltonian H1 (coupling) */
    /*struct Coupling {
        int i; int j; float H1;
    };
    std::vector<Coupling> couplings(0);*/

    int couplings = 0;
    for (auto i = 0; i < N; i++) {
        for (auto j = i + 1; j < N; j++) {
            if (fabsf(Hamiltonian_i[i][j]) > non->couplingcut) {
                //couplings.push_back({ i, j, Hamiltonian_i[i][j] });
                H1[couplings] = Hamiltonian_i[i][j];
                row[couplings] = i;
                col[couplings] = j;
                couplings++;
            }
        }
    }

    af::array H1_G(N * N / 2, H1);
    af::array row_G(N * N / 2, row);
    af::array col_G(N * N / 2, col);
    af::array H0_G(N, N, H0[0]);
    af::array U_G = af::exp(af::complex(H0_G) * 0.5 * f);
    af::array c_G(N, N, c[0]);
    af::array f_G = af::constant(f, 1);
    af::array sqrt2_G = af::constant(sqrt2, 1);

    /* Exponentiate diagonal [U=exp(-i/2h H0 dt)] */
    /*for (int a = 0; a < N2; a++) {
        re_U[a] = cosf(0.5f * H0[a] * f);
        im_U[a] = -sinf(0.5f * H0[a] * f);
    }*/

    for (int i = 0; i < m; i++) {

        /* Multiply on vector first time */
        /*for (int a = 0; a < N2; a++) {
            ocr[a] = cr[a] * re_U[a] - ci[a] * im_U[a];
            oci[a] = cr[a] * im_U[a] + ci[a] * re_U[a];
        }*/

        af::array oc_G = c_G * U_G;

        for (int j = 0; j < couplings; j++) {
            af::array j_G = af::constant(j, 1);
            auto a = row[j];
            auto b = col[j];

            af::array a_G = row_G(j_G);
            af::array b_G = col_G(j_G);

            gfor(af::seq k, a) {
                auto si_G = -af::sin(H1_G(j_G) * f_G);
                auto co_G = af::sqrt(1 - si_G * si_G);

                af::array x_val = oc_G(k, a_G);
                oc_G(k, a_G) *= co_G;
                oc_G(k, a_G) += si_G * oc_G(k, b_G) * af::cfloat(0, 1);
                oc_G(k, b_G) *= co_G;
                oc_G(k, b_G) += si_G * x_val * af::cfloat(0, 1);
            }

            gfor(af::seq k, a, b) {
                auto si_G = -af::sin(H1_G(j_G) * f_G);
                auto co_G = af::sqrt(1 - si_G * si_G);

                af::array x_val = oc_G(a_G, k);
                oc_G(a_G, k) *= co_G;
                oc_G(a_G, k) += si_G * oc_G(k, b_G) * af::cfloat(0, 1);
                oc_G(k, b_G) *= co_G;
                oc_G(k, b_G) += si_G * x_val * af::cfloat(0, 1);
            }

            gfor(af::seq k, b, N - 1) {
                auto si_G = -af::sin(H1_G(j_G) * f_G);
                auto co_G = af::sqrt(1 - si_G * si_G);

                af::array x_val = oc_G(a_G, k);
                oc_G(a_G, k) *= co_G;
                oc_G(a_G, k) += si_G * oc_G(b_G, k) * af::cfloat(0, 1);
                oc_G(b_G, k) *= co_G;
                oc_G(b_G, k) += si_G * x_val * af::cfloat(0, 1);
            }

            
            /*
            auto a = coupling.i;
            auto b = coupling.j;

            for(auto c = 0; c < N; c++) {
                auto si = c == a || c == b ? -sinf(coupling.H1 * f * sqrt2) : -sinf(coupling.H1 * f);
                auto co = sqrtf(1 - si * si);

                auto x = a < c ? oc_G(a, c) : oc_G(c, a);
                auto y = b < c ? oc_G(b, c) : oc_G(c, b);
                af::array x_val = x;

                x *= co; x += si * y * af::cfloat(0, 1);
                y *= co; y += si * x_val * af::cfloat(0, 1);
            }*/
        }

        //ocr = ocr_G.host<float>();
        //oci = oci_G.host<float>();

        /* Account for couplings */
        /* Loop over couplings */
        /*for (int k = 0; k < kmax; k++) {
            int a = col[k];
            int b = row[k];
            float J = H1[k] * f;

            /* Loop over wave functions <ca|Hab|cb> and <cb|Hba|ca> #1#
            // TODO speedup
            for (int c = 0; c < N; c++) {
                float si = (c == a || c == b) ? -sinf(J * sqrt2) : -sinf(J);

                float co = sqrtf(1 - si * si);
                int index1 = Sindex(a, c, N), index2 = Sindex(c, b, N);

                oc[a][c] *= co;
                oc[a][c] += i * si * oc[b][c];
                auto x = oc[a][c];
                oc[b][c] *= co;
                oc[b][c] += i * si * x;

                float cr1 = co * ocr[index1] - si * oci[index2];
                float ci1 = co * oci[index1] + si * ocr[index2];
                float cr2 = co * ocr[index2] - si * oci[index1];
                float ci2 = co * oci[index2] + si * ocr[index1];
                ocr[index1] = cr1, oci[index1] = ci1, ocr[index2] = cr2, oci[index2] = ci2;
            }
        }*/

        /* Multiply on vector second time */
        c_G = oc_G * U_G;

        /*for (int a = 0; a < N2; a++) {
            cr[a] = ocr[a] * re_U[a] - oci[a] * im_U[a];
            ci[a] = ocr[a] * im_U[a] + oci[a] * re_U[a];
        }*/
    }

    // Copy data back to host
    c_G.host(c[0]);
    convert2Dto1Dsymm(c, c_1D, N);

    for (auto i = 0; i < N2; i++) {
        cr_1D[i] = c_1D[i].real;
        ci_1D[i] = c_1D[i].imag;
    }
    
    delete[] H0; delete[] H1; delete[] col; delete[] row;
}
