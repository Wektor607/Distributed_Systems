#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <signal.h>
#include <mpi-ext.h>

MPI_Comm ACTIVE_COMM = MPI_COMM_WORLD, cannon_comm;

int Nl;
double *A, *B, *C;
int global_rank, global_size;
int coords[2];
int left, right, up, down;
int ndims = 2;
int dims[2];
int periods[2];

static void data_save()
{
    if (global_rank == global_size - 2) {
        FILE* f = fopen("data", "wb");
        fwrite(&A[0], sizeof(double),  Nl * Nl, f);
        fwrite(&B[0], sizeof(double),  Nl * Nl, f);
        fwrite(&C[0], sizeof(double),  Nl * Nl, f);
        fclose(f);
    }
    MPI_Barrier(ACTIVE_COMM);
}

static void data_load()
{
    if (global_rank == global_size - 2) {
        FILE* f = fopen("data", "rb");

        fread(&A[0], sizeof(double),  Nl * Nl, f);
        fread(&B[0], sizeof(double),  Nl * Nl, f);
        fread(&C[0], sizeof(double),  Nl * Nl, f);
        fclose(f);
        printf("Proc %d\n", global_rank);
    }

    MPI_Barrier(ACTIVE_COMM);
}

static void verbose_errhandler(MPI_Comm* pcomm, int* perr, ...) {
    MPI_Comm comm = *pcomm;
    int err = *perr;
    char errstr[MPI_MAX_ERROR_STRING];
    int i, rank, size, nf, len, eclass;
    MPI_Group group_c, group_f;
    int *ranks_gc, *ranks_gf;
    MPI_Error_class(err, &eclass);
    if( MPIX_ERR_PROC_FAILED != eclass ) {
        MPI_Abort(comm, err);
    }
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPIX_Comm_failure_ack(comm);
    MPIX_Comm_failure_get_acked(comm, &group_f);
    MPI_Group_size(group_f, &nf);
    MPI_Error_string(err, errstr, &len);
    printf("Rank %d / %d: Notified of error %s. %d found dead: { ", rank, size, errstr, nf);
    ranks_gf = (int*)malloc(nf * sizeof(int));
    ranks_gc = (int*)malloc(nf * sizeof(int));
    MPI_Comm_group(comm, &group_c);
    for(i = 0; i < nf; i++) {
        ranks_gf[i] = i;
    }
    MPI_Group_translate_ranks(group_f, nf, ranks_gf,
    group_c, ranks_gc);
    for(i = 0; i < nf; i++) {
        printf("%d ", ranks_gc[i]);
    }
    printf("}\n");

    MPIX_Comm_shrink(comm, &ACTIVE_COMM);
    MPI_Comm_rank(ACTIVE_COMM, &global_rank);

    MPI_Cart_create(ACTIVE_COMM, ndims, dims, periods, 0, &cannon_comm);

    MPI_Cart_coords(cannon_comm, global_rank, 2, coords);
    MPI_Cart_shift(cannon_comm, 1, 1, &left, &right);
    MPI_Cart_shift(cannon_comm, 0, 1, &up, &down);

    data_load();
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("ERROR: Please enter NODE_NUMBERA: \n./run NODE_NUMBERS\n");
        return -1;
    }

    double start, end;
    double *buf, *tmp;
    
    long long int N = atoi(argv[1]);
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &global_size);
    
    MPI_Errhandler errh;
    MPI_Comm_create_errhandler(verbose_errhandler, &errh);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, errh);
    
    
    dims[0] = 0;
    dims[1] = 0;
    periods[0] = 1;
    periods[1] = 1;
    MPI_Dims_create(global_size-1, 2, dims);
    if(dims[0] != dims[1]) 
    {
        if(global_rank == 0) 
        {
            printf("The number of processors must be a square.\n");
        }
        MPI_Finalize();
        return 0;
    }
    Nl = N / dims[0];
    A = (double*)malloc(Nl * Nl * sizeof(double));
    B = (double*)malloc(Nl * Nl * sizeof(double));
    buf = (double*)malloc(Nl *Nl * sizeof(double));
    C = (double*)calloc(Nl * Nl, sizeof(double));
    for(int i=0; i<Nl; i++)
    {
        for(int j=0; j<Nl; j++) 
        {
            A[i*Nl+j] = global_rank;//5 - (int)( 10.0 * rand() / ( RAND_MAX + 1.0 ) );
            B[i*Nl+j] = global_rank;//5 - (int)( 10.0 * rand() / ( RAND_MAX + 1.0 ) );
            C[i*Nl+j] = 0.0;
        }
    }
    

    MPI_Comm new_comm;
    MPI_Comm_split(MPI_COMM_WORLD, global_rank != global_size-1, global_rank, &new_comm);
    if (global_rank != global_size-1) 
    {
        
        MPI_Cart_create(new_comm, ndims, dims, periods, 0, &cannon_comm);

        MPI_Cart_coords(cannon_comm, global_rank, 2, coords);
        MPI_Cart_shift(cannon_comm, 1, coords[0], &left, &right);
        MPI_Cart_shift(cannon_comm, 0, coords[1], &up, &down);

        if (coords[0]) {
            MPI_Sendrecv(A, Nl * Nl, MPI_DOUBLE, left, 1, buf, Nl * Nl, MPI_DOUBLE, right, 1, cannon_comm, &status);
            tmp = buf; 
            buf = A; 
            A = tmp;
        }
        if (coords[1]) {
            MPI_Sendrecv(B, Nl * Nl, MPI_DOUBLE, up, 2, buf, Nl * Nl, MPI_DOUBLE, down, 2, cannon_comm, &status);
            tmp = buf; 
            buf = B; 
            B = tmp;
        }

        MPI_Cart_shift(cannon_comm, 1, 1, &left, &right);
        MPI_Cart_shift(cannon_comm, 0, 1, &up, &down);
    }
    start = MPI_Wtime();

    for(int shift=0; shift < dims[0]; shift++) 
    {
        for(int i=0; i<Nl; i++)
        {
            for(int k=0; k<Nl; k++)
            {
                for(int j=0; j<Nl; j++)
                {
                    C[i*Nl+j] += A[i*Nl+k]*B[k*Nl+j];
                }
            }
        }
        data_save();
        if(shift == dims[0] / 2)
        {
            if (global_rank == global_size - 2) {
                raise(SIGKILL);
            }
        }
        MPI_Barrier(ACTIVE_COMM);
        if(shift == dims[0] - 1) 
        {
            break;
        }

        if (global_rank != global_size-1) 
        {
            MPI_Sendrecv(A, Nl * Nl, MPI_DOUBLE, left, 1, buf, Nl * Nl, MPI_DOUBLE, right, 1, cannon_comm, &status);
            tmp = buf; 
            buf = A; 
            A = tmp;
            MPI_Sendrecv(B, Nl * Nl, MPI_DOUBLE, up, 2, buf, Nl * Nl, MPI_DOUBLE, down, 2, cannon_comm, &status);
            tmp = buf; 
            buf = B; 
            B = tmp;
        }
    }
    MPI_Barrier(ACTIVE_COMM);
    end = MPI_Wtime();
    if(global_rank == 0) 
    {
        printf("Time: %.4fs\n", end-start);
    }
    
    free(A); 
    free(B); 
    free(buf); 
    free(C);

    MPI_Finalize();
    return 0;
}