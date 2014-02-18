//
//  main.cpp
//

#include "Matrix.hpp"

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <mpi.h>

using namespace std;

struct data{
    float value;
    int   index;
};

// Inverser la matrice par la méthode de Gauss-Jordan; implantation séquentielle.
void invertSequential(Matrix& iA) {

    // vérifier que la matrice est carrée
    assert(iA.rows() == iA.cols());
    // construire la matrice [A I]
    MatrixConcatCols lAI(iA, MatrixIdentity(iA.rows()));

    // traiter chaque rangée
    for (size_t k=0; k<iA.rows(); ++k) {
        // trouver l'index p du plus grand pivot de la colonne k en valeur absolue
        // (pour une meilleure stabilité numérique).
        size_t p = k;
        double lMax = fabs(lAI(k,k));
        for(size_t i = k; i < lAI.rows(); ++i) {
            if(fabs(lAI(i,k)) > lMax) {
                lMax = fabs(lAI(i,k));
                p = i;
            }
        }
        // vérifier que la matrice n'est pas singulière
        if (lAI(p, k) == 0) throw runtime_error("Matrix not invertible");

        // échanger la ligne courante avec celle du pivot
        if (p != k) lAI.swapRows(p, k);

        double lValue = lAI(k, k);
        for (size_t j=0; j<lAI.cols(); ++j) {
            // On divise les éléments de la rangée k
            // par la valeur du pivot.
            // Ainsi, lAI(k,k) deviendra égal à 1.
            lAI(k, j) /= lValue;
        }

        // Pour chaque rangée...
        for (size_t i=0; i<lAI.rows(); ++i) {
            if (i != k) { // ...différente de k
                // On soustrait la rangée k
                // multipliée par l'élément k de la rangée courante
                double lValue = lAI(i, k);
                lAI.getRowSlice(i) -= lAI.getRowCopy(k)*lValue;
            }
        }
    }

    // On copie la partie droite de la matrice AI ainsi transformée
    // dans la matrice courante (this).
    for (unsigned int i=0; i<iA.rows(); ++i) {
        iA.getRowSlice(i) = lAI.getDataArray()[slice(i*lAI.cols()+iA.cols(), iA.cols(), 1)];
    }
}

// Inverser la matrice par la méthode de Gauss-Jordan; implantation MPI parallèle.
void invertParallel(Matrix& iA) {

    int myrank, ranksize;
    ranksize = MPI::COMM_WORLD.Get_size(); // nombre de processus
    myrank = MPI::COMM_WORLD.Get_rank(); // numero du processus courant (me)

    // vérifier que la matrice est carrée
    assert(iA.rows() == iA.cols());
    // construire la matrice [A I]
    MatrixConcatCols lAI(iA, MatrixIdentity(iA.rows()));

    //for (size_t k=0; k<iA.rows(); ++k) {
    for (size_t k=0; k<1; ++k) {

        // on trouve le q localement (par processus)
        double lMax = 0.0;
        for(size_t i = k; i < lAI.rows(); ++i) {
            if( (i % ranksize) == myrank) {

                if(fabs(lAI(i,k)) > lMax) {
                    lMax = fabs(lAI(i,k));
                }

            }
        }

        //cout << "Process " << myrank <<  " a essayé " << i << " " << k << lAI(i,k) <<endl;
        cout << "Process " << myrank <<  " a trouvé " << lMax <<endl;

        data in, out;
        in.index = myrank;
        in.value = lMax;

        //float gMax = 0.0;
        MPI_Allreduce(&in, &out, iA.rows(), MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        cout << "Max trouvé " << out.value << endl;

    }

    MPI::COMM_WORLD.Barrier();

}

// Multiplier deux matrices.
Matrix multiplyMatrix(const Matrix& iMat1, const Matrix& iMat2) {

    // vérifier la compatibilité des matrices
    assert(iMat1.cols() == iMat2.rows());
    // effectuer le produit matriciel
    Matrix lRes(iMat1.rows(), iMat2.cols());
    // traiter chaque rangée
    for(size_t i=0; i < lRes.rows(); ++i) {
        // traiter chaque colonne
        for(size_t j=0; j < lRes.cols(); ++j) {
            lRes(i,j) = (iMat1.getRowCopy(i)*iMat2.getColumnCopy(j)).sum();
        }
    }
    return lRes;
}


int main(int argc, char** argv) {

    srand((unsigned)time(NULL));

    int myrank, ranksize;

    // get input from user
    unsigned int lS = 5;
    if (argc == 2) {
        lS = atoi(argv[1]);
    }

    // on initialise une matrix sequential et parallel pour comparaison
    Matrix mainMatrixSequential = Matrix(lS, lS);
    Matrix mainMatrixParallel = Matrix(lS, lS);
    MatrixRandom lAPara(lS, lS);

    // section sequentielle
    MatrixRandom lA(lS, lS);
    //cout << "Matrice random:\n" << lA.str() << endl;

    mainMatrixSequential = lA;

    invertSequential(mainMatrixSequential);
    //cout << "Matrice inverse:\n" << mainMatrixSequential.str() << endl;

    Matrix lRes = multiplyMatrix(lA, mainMatrixSequential);
    //cout << "Produit des deux matrices:\n" << lRes.str() << endl;

    // only rank 0 initialize the matrix
    //cout << "Erreur: " << lRes.getDataArray().sum() - lS << endl;


    // section parallele
    MPI::Init();

    ranksize = MPI::COMM_WORLD.Get_size();
    myrank = MPI::COMM_WORLD.Get_rank();

    mainMatrixParallel = lAPara; // init parallel matrix to compare both
    // inversion de la matrice parallel
    invertParallel(mainMatrixParallel);

    if(myrank == 0) {
        cout << "\nMatrice inverse parallele:\n" << mainMatrixParallel.str() << endl;
    }

    MPI::Finalize();

    return 0;
}

