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
    int n = iA.rows();

    // vérifier que la matrice est carrée
    assert(iA.rows() == iA.cols());
    // construire la matrice [A I]
    MatrixConcatCols lAI(iA, MatrixIdentity(iA.rows()));

    //for (size_t k=0; k < 1; ++k) {
    for (size_t k=0; k < lAI.rows(); ++k) {

        // on trouve le q localement (par processus)
        double lMax = 0.0;
        size_t q = k;
        for(size_t i = k; i < lAI.rows(); ++i) {
            if( (i % ranksize) == myrank) {

                if(fabs(lAI(i,k)) > lMax) {
                    lMax = fabs(lAI(i,k));
                    q = i;
                }
            }
        }

        //cout << "Process " << myrank <<  " a essayé " << i << " " << k << lAI(i,k) <<endl;
        //cout << "Process " << myrank <<  " a trouvé " << lMax <<endl;

        data in, out;
        in.index = q;
        in.value = lMax;

        MPI::COMM_WORLD.Allreduce(&in, &out, ranksize, MPI_FLOAT_INT, MPI_MAXLOC);
        //cout << "Max trouvé " << out.value << " " << out.index << endl;
        q = out.index;
        int root = q%ranksize;

        // broadcast a tous les processus les elements de k a n-1 de l'indice q trouve
        MPI::COMM_WORLD.Bcast(&lAI(q,0), lAI.cols(), MPI::DOUBLE, root);

        // vérifier que la matrice n'est pas singulière
        if (lAI(q, k) == 0) throw runtime_error("Matrix not invertible");

        // on swap la ligne q avec la ligne k
        if (q != k) lAI.swapRows(q, k);

        // on normalise la ligne k afin que l'element (k,k) soit egale a 1
        double lValue = lAI(k, k);
        for (size_t j=0; j<lAI.cols(); ++j) {
            lAI(k, j) /= lValue;
        }

        //// Pour chaque rangée...
        for (size_t i=0; i<lAI.rows(); ++i) {
            if( (i % ranksize) == myrank) {
                if (i != k) { // ...différente de k
                    // On soustrait la rangée k
                    // multipliée par l'élément k de la rangée courante
                    double lValue = lAI(i, k);
                    //cout << "process " << myrank << " soustrait la rangée " << i << " par le k " << lValue << endl;
                    lAI.getRowSlice(i) -= lAI.getRowCopy(k)*lValue;
                }
            }
        }

        for(size_t i = 0; i < lAI.rows(); ++i) {
            MPI::COMM_WORLD.Bcast(&lAI(i,0), lAI.cols(), MPI::DOUBLE, i%ranksize);
        }

    }


    //cout << "\nMatrice parallele du processus " << myrank << "\n" << lAI.str() << endl;

     //now we want to gather all the rows to process 0
    //for(size_t i = 0; i<lAI.rows(); ++i) {
        //if( (i % ranksize) == myrank && myrank != 0) {
            //cout << "Rnk " << myrank << " envoie ligne " << i << endl;
            //MPI::COMM_WORLD.Send(&lAI(i,0), lAI.cols(), MPI_DOUBLE, 0, 1234);
        //}
        //else if( (i % ranksize) != myrank && myrank == 0) {
            //cout << "Rnk " << myrank << " recoit ligne " << i << endl;
            //MPI::COMM_WORLD.Recv(&lAI(i,0), lAI.cols(), MPI_DOUBLE, MPI_ANY_SOURCE, 1234);
        //}
    //}



    //if(myrank == 1) {
        cout << "\nMatrice parallele du processus " << myrank << "\n" << lAI.str() << endl;
    //}
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



    //
    // section sequentielle
    //
    MatrixRandom lA(lS, lS);
    //cout << "Matrice random:\n" << lA.str() << endl;
    mainMatrixSequential = lA;
    invertSequential(mainMatrixSequential);
    //cout << "Matrice inverse:\n" << mainMatrixSequential.str() << endl;
    Matrix lRes = multiplyMatrix(lA, mainMatrixSequential);
    //cout << "Produit des deux matrices:\n" << lRes.str() << endl;
    //cout << "Erreur: " << lRes.getDataArray().sum() - lS << endl;

    //
    // section parallele
    //
    MPI::Init();

    ranksize = MPI::COMM_WORLD.Get_size();
    myrank = MPI::COMM_WORLD.Get_rank();

    if ( myrank == 0 ) {
        // seul le root creer la matrice random, puis on a la broadcast
        MatrixRandom lAPara(lS, lS);
        mainMatrixParallel = lAPara;
    }
    // on bcast la matrice random nouvellement creer
    MPI::COMM_WORLD.Bcast(&mainMatrixParallel(0,0), lS*lS, MPI::DOUBLE, 0);

    // inversion de la matrice parallel
    invertParallel(mainMatrixParallel);

    if(myrank == 0) {
        cout << "\nMatrice inverse parallele:\n" << mainMatrixParallel.str() << endl;
    }

    MPI::Finalize();

    return 0;
}

