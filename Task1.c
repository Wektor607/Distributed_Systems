#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <unistd.h>

#define MPI_TAG 25

int world_rank = 0;
int *query, *query_head, *query_tail;
int marker_rank;

int flag;
int *processesRemaining;

MPI_Win wintable;

MPI_Aint winsize;
int windisp;

double start_time, end_time;

char *critical_file = "critical.txt";

void move_query()
{
    query_head++;
    if (query_head > query_tail)
    {
        query_head = query;
        query_tail = query;
        *query = -1;
    }
}

void add_query(int rank)
{
    *query_tail = rank;
    query_tail++;
}

void check_out()
{
    if (processesRemaining[0] == 0)
    {
        remove(critical_file);
        end_time = MPI_Wtime();
        printf("Работа окончена! Это заняло %f секунд\n", end_time-start_time);
        exit(0);
    }
}

void critical_section()
{
    MPI_Win_shared_query(wintable, 0, &winsize, &windisp, &processesRemaining);

    processesRemaining[0] = processesRemaining[0] - 1;

    printf("Текущий процесс: %d, осталось: %d процессов!\n", world_rank, processesRemaining[0]);
    FILE* fp;
    if (fp = fopen(critical_file, "r"))
    {
        fclose(fp);
        printf("Ошибка!: критический файл уже существует! Вызвана процессом: %d\n",world_rank);
        exit(1);
    } 
    else 
    {
        if(fp = fopen(critical_file, "w"))
        {
            fclose(fp);
            printf("Процесс %d in!\n", world_rank);
            check_out();
            printf("Процесс %d out!\n", world_rank);
            remove(critical_file);
            return;
        } 
        else 
        {
            printf("Ошибка! Не удалось открыть файл на запись!\n");
            exit(1);
        }
    }
}

void send_marker(int whither)
{
    int temp = -1;
    MPI_Request request;
    MPI_Isend(&temp, 1, MPI_INT, whither, MPI_TAG, MPI_COMM_WORLD, &request);
    MPI_Request_free(&request);
}

void send_request(int marker_rank)
{
    MPI_Request request;
    MPI_Isend(&world_rank, 1, MPI_INT, marker_rank, MPI_TAG, MPI_COMM_WORLD, &request);
    MPI_Request_free(&request);
}

void accept_marker()
{
    marker_rank = world_rank;
    if (*query_head == world_rank)
    {
        move_query();
        return;
    }
    if (*query_head== -1)
    {
        return;
    }
    marker_rank = *query_head;
    send_marker(*query_head);
    move_query();
    if (*query_head != -1)
    {
        send_request(marker_rank);
    }
}

void accept_request(int rank) 
{
    add_query(rank);

    if (marker_rank == world_rank) 
    {
        marker_rank = *query_head;
        send_marker(*query_head);
        move_query();
        if (*query_head != -1)
        {
            send_request(marker_rank);
        }
    } 
    else 
    {
        send_request(marker_rank); 
        // отправили свой ранк процессу с маркером
    }
}

void wait_marker()
{
    int buffer = 0;
    MPI_Status status;
    MPI_Request request;
    MPI_Irecv(&buffer, 1, MPI_INT, MPI_ANY_SOURCE, MPI_TAG, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, &status);
    if (buffer==-1)
    {
        accept_marker();
        return;
    }
    accept_request(buffer);

    wait_marker();
}

void check_query()
{
    int buffer = 0;
    MPI_Status status;
    MPI_Request request;
    MPI_Irecv(&buffer, 1, MPI_INT, MPI_ANY_SOURCE, MPI_TAG, MPI_COMM_WORLD, &request); 

    int flag = 0, counter = 1024;
    MPI_Test(&request, &flag, &status);
    while((!flag) && (counter>0)) 
    {
        sleep(1);
        MPI_Test(&request, &flag, &status);
        counter--;
    }
    if (counter == 0)
    {
        return;
    }

    if (buffer == -1)
    {
        accept_marker();
    } 
    else 
    {
        accept_request(buffer);
    }

    check_query();
}

int main(int argc, char ** argv){
    //Инициализируем MPI, получаем необходимые переменные, инициализируем очередь
    int status = MPI_Init(&argc, &argv);
    if (status != MPI_SUCCESS)
    {
        return status;
    }
    start_time = MPI_Wtime();
    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    query = (int*)malloc(world_size*sizeof(int)*1024);
    for (int i=0; i<world_size; i++)
    {
        query[i] = -1;
    }
    query_head = query;
    query_tail = query;

    //Настраиваем общую память
    int tablesize = 1;
    MPI_Comm nodecomm;

    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &nodecomm);
    MPI_Comm_size(nodecomm, &world_size);
    MPI_Comm_rank(nodecomm, &world_rank);

    int localtablesize = 0;
    if (world_rank == 0)
    { 
        localtablesize = 1;
    }

    int *model;
    int *localtable;
    MPI_Win_allocate_shared(localtablesize*sizeof(int), sizeof(int), MPI_INFO_NULL, nodecomm, &localtable, &wintable);
    MPI_Win_get_attr(wintable, MPI_WIN_MODEL, &model, &flag);
    if (flag != 1)
    {
        printf("Attribute MPI_WIN_MODEL not defined\n");
    } 
    else 
    {
        if (MPI_WIN_UNIFIED == *model)
        {
            if (world_rank == 0) printf("Memory model is MPI_WIN_UNIFIED\n");
        } 
        else 
        {
            if (world_rank == 0) printf("Memory model is *not* MPI_WIN_UNIFIED\n");
            MPI_Finalize();
            return -1;
        }
    }

    processesRemaining = localtable;

    if (world_rank != 0)
    {
        MPI_Win_shared_query(wintable, 0, &winsize, &windisp, &processesRemaining);
    }

    MPI_Win_fence(0, wintable);

    if (world_rank == 0)
    {
        processesRemaining[0] = world_size;
    }

    MPI_Win_fence(0, wintable);  

    marker_rank = (world_rank == 0) ? 0 : (world_rank - 1) / 2;

    //Используем барьерную синхронизацию
    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank != marker_rank) 
    {
        accept_request(world_rank);
        wait_marker();
    }

    critical_section();

    check_query();

    MPI_Finalize();
    return 0;
}