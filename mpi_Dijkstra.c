#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <memory.h>

//0th rank -> Main Process
//Total number of vertices are sent and received using tag 0
//The offset vector is sent and received using tag 1
//The number of nodes each process needs to handle stored in localNumOfElements is sent and received using tag 2
//The total number of weights sent to a process is sent and received using tag 3
//The actual weights for a particular process is sent and received using tag 4

int rowMajor(int row, int col, int n) {
    return row * n + col;
}

static void calculateOffset(int ** offset, int ** localNumOfElements, int numberOfProcessors, int size){
    int local, reminder;
    //5 10
    if (size < numberOfProcessors){
        local = 1;
        //[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        *localNumOfElements = malloc(numberOfProcessors * sizeof(**localNumOfElements)); //sizeof(int)
        int i = 0;
        for (; i < size; i++){
            *(*localNumOfElements + i) = local;
        }
        for (; i < numberOfProcessors;i++){
            *(*localNumOfElements + i) = 0;
        }
    } else {
        //7 3
        local = size / numberOfProcessors; 
        reminder = size % numberOfProcessors; 

        // localNumOfElement = [2, 2, 3]
        *localNumOfElements = malloc(numberOfProcessors * sizeof(**localNumOfElements));
        for (int i = 0; i < numberOfProcessors; i++) {
            *(*localNumOfElements + i) = local;
        }
        *(*localNumOfElements + numberOfProcessors - 1) += reminder;
    }
    *offset = malloc(numberOfProcessors * sizeof(**offset));
    //offset = [0,2,4]
    for (int i = 0; i < numberOfProcessors; i++) {
        *(*offset + i) = 0;
        for (int j = 0; j < i; j++) {
            *(*offset + i) += *(*localNumOfElements + j);
        }
    }
}
static void load(char const *const filename, int *const np, float **const ap, int numberOfProcessors, int ** offset, int ** localNumOfElements, int rank) {
    int n;
    float *a = NULL;
    //only Main Process will run in the if block
    if (rank == 0) {
        int i, j, k, ret;
        FILE *fp = NULL;
        fp = fopen(filename, "r");

        fscanf(fp, "%d", &n);
        //printf("n is %d\n", n);

        calculateOffset(offset, localNumOfElements, numberOfProcessors, n);

        a = malloc(n * *(*localNumOfElements) * sizeof(*a));

        for (j = 0; j < *(* localNumOfElements) * n; ++j) {
            fscanf(fp, "%f", &a[j]);
        }
        *ap = a;

        for (i = 1; i < numberOfProcessors; ++i) {
            a = malloc(n * *(*localNumOfElements + i) * sizeof(*a));

            //communicate everything to the Global Communicator
            MPI_Send(&n, 1, MPI_INTEGER, i, 0, MPI_COMM_WORLD);
            MPI_Send(*offset, numberOfProcessors, MPI_INTEGER, i, 1, MPI_COMM_WORLD);
            MPI_Send(*localNumOfElements, numberOfProcessors, MPI_INTEGER, i, 2, MPI_COMM_WORLD);

            //reading weights from file //every process reads diff weigths
            for (j = 0; j < *(* localNumOfElements + i) * n; ++j) {
                fscanf(fp, "%f", &a[j]); 
            }

            MPI_Send(&j, 1, MPI_INTEGER, i, 3, MPI_COMM_WORLD);
            MPI_Send(a, j, MPI_FLOAT, i, 4, MPI_COMM_WORLD);
            free(a);
        }
        fclose(fp);
    } else {
        int count;
        MPI_Recv(&n, 1, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        *offset = malloc(numberOfProcessors * sizeof(**offset));
        MPI_Recv(*offset, numberOfProcessors, MPI_INTEGER, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        *localNumOfElements = malloc(numberOfProcessors * sizeof(**localNumOfElements));
        MPI_Recv(*localNumOfElements, numberOfProcessors, MPI_INTEGER, 0, 2, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Recv(&count, 1, MPI_INTEGER, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        a = malloc(count * sizeof(*a));
        MPI_Recv(a, count, MPI_FLOAT, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        *ap = a;
    }

    *np = n;
    MPI_Barrier(MPI_COMM_WORLD);
}


static void dijkstra(int const source, int const n, float const *const a, float **const result, int rank, int * offset, int * localNumOfElements, int numberOfProcessors) {
    struct float_int {
        float distance;
        int u;
    } min;
    
    int i, j, k, sourceBlock = 0;
    char *visited = NULL;
    float *resultVector = NULL;
    float * localResult = NULL;


    //[0,0,0,0,0,0,0]
    visited = calloc(n, sizeof(*visited));

    //[0,0,0,0,0,0,0]
    resultVector = malloc(n * sizeof(*resultVector));

    //[0,0,0,0,0,0,0]
    localResult = malloc(n * sizeof(*resultVector));

    for (i = 0; i < numberOfProcessors; i++){
        if (source < offset[i]){
            sourceBlock = i - 1;
            break;
        }
    }
    
    if (rank == sourceBlock) {
        for (i = 0; i < n; ++i) {
            resultVector[i] = a[i + n * (source - offset[sourceBlock])];
            
        }
    }
    MPI_Bcast(resultVector, n, MPI_FLOAT, sourceBlock, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    //printf("Vertex %d: Local min is: %.1f\n", rank, resultVector[rank]);

    visited[source] = 1;
    min.u = -1; 

    for (i = 1; i < n; ++i) {
        min.distance = INFINITY;
        for (j = 0; j < n; ++j) {
            if (!visited[j] && resultVector[j] < min.distance) {
                min.distance = resultVector[j];
                min.u = j;
            }
            localResult[j] = resultVector[j];
        }
       
        visited[min.u] = 1;
        for (j = 0; j < localNumOfElements[rank]; j++){
            if (visited[j + offset[rank]]){
                continue;
            }
            if (a[rowMajor(j,min.u,n)] + min.distance < localResult[j + offset[rank]]){
                localResult[j + offset[rank]] = a[rowMajor(j,min.u,n)]  + min.distance;
            }
            //printf("Vertex %d: Local min is: %.1f\n", j, localResult[j + offset[rank]]);
        }
        MPI_Allreduce(localResult, resultVector, n, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    free(visited);

    *result = resultVector;
}

static void print_to_file(char const *const filename, int const n, float const *const numbers) {
    int i;
    FILE *fout;

    if (NULL == (fout = fopen(filename, "w"))) {
        fprintf(stderr, "error opening '%s'\n", filename);
        abort();
    }

    for (i = 0; i < n; ++i) {
        fprintf(fout, "%10.4f\n", numbers[i]);
    }

    fclose(fout);
}

int main(int argc, char **argv) {
    int n, numberOfProcessors, rank;
    double ts, te;
    float *a = NULL, *result = NULL;
    int * offset = NULL, *localNumOfElements = NULL;

    if (argc < 4) {
        printf("Invalid number of arguments.\nUsage: dijkstra <graph> <source> <output_file>.\n");
        return EXIT_FAILURE;
    }


    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcessors);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    load(argv[1], &n, &a, numberOfProcessors, &offset, &localNumOfElements, rank);

    MPI_Barrier(MPI_COMM_WORLD);
    ts = MPI_Wtime();
    dijkstra(atoi(argv[2]), n, a, &result, rank, offset, localNumOfElements, numberOfProcessors);
    te = MPI_Wtime();

    if (rank == 0) {
        printf("Operation Time: %0.04fs\n", te - ts);
        print_to_file(argv[3], n, result);
    }
    free(a);
    free(result);
    MPI_Finalize();
    return EXIT_SUCCESS;
}