/****************************************************************************
 * FILE: mpi.c
 * DESCRIPTION:
 *   simple 2d temperature distribution MPI solution
 * MODIFIED BY: Rita Borgo
 * CONVERTED TO MPI BY: Michael Waterworth - Dom Chim
 * Last Revised: 8/12/14 Michael Waterworth - Dom Chim
 * A couple of things to be wary of:
 * * X dimension is vertical and goes first.
 * * Y dimension is horizontal and goes second.
 * Run on HPCW GL Cluster
 * Needs to have a output/ folder to catch all the files flying out
 ****************************************************************************/
#include "mpi.h"
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#define XDIM 100
#define YDIM 100
#define MASTER 0


struct Params
{
    double cx; /* sampling size along X */
    double cy; /* sampling size along Y */
    int nts; /* timesteps */
}params = {0.1,0.1,100};


void initdata(int nx, int ny, double *u1);
void update(int nx, int ny, double *u1, double *u2);
void split(double *fMat, double *split, int startRow, int endRow);
void combine(double *segMatrix, double *tarMatrix, int startRow,int endRow);
void prtdata(int nx, int ny, int ts, double *u1, char* fname);


int main(int argc, char *argv[])
{   
    clock_t start = clock(), diff;
    int numtasks, rank;
    MPI_Status status;

    MPI_Init(&argc, &argv);/*Initialise*/
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);/*Get number of nodes*/
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    if(numtasks <= 1){
        printf("You need at least 2 nodes!\n");
        return 1;
    }

    /*Find the size of matrices sent to other nodes.*/
    double splitSizeD = (double)XDIM / (double)(numtasks-1);
    int splitSizeI = (int) ceil(splitSizeD);

    splitSizeI += 2;/*Pad it out for above and below values*/

    if(splitSizeI > XDIM)
    {
        splitSizeI = XDIM;
    }

    /*
    segmented (broken) arrays
    These are used on nodes > 1 to do the calculations.
    */
    double *broken_u_a = (double*)calloc(splitSizeI * YDIM, sizeof(double));/*old_u*/
    double *broken_u_b = (double*)calloc(splitSizeI * YDIM, sizeof(double));/*new_u*/

    if(rank == MASTER)
    {
        /*
        Full arrays
        Only available on the MASTER node.
        Holds the full Matrix and used to coordinate
        */
        double *full_u_a = (double*)calloc(XDIM * YDIM, sizeof(double));/*old_u*/
        double *full_u_b = (double*)calloc(XDIM * YDIM, sizeof(double));/*new_u*/
        
        int ix, iy,it; /* iterators */
        char buf[256], *str_end=".csv"; /*file formatting*/
        
        /* Iterate over all timesteps and create final output file (e.g. timestep = nts) */
        int splitSize = splitSizeI; /* 11 nodes */

        /*
        Size of split data sent off to nodes.
        Take size of split & add 2 (1up, 1down)
        Iterate counting size of split, not these extra rows.
        */
        int splitSizeReal = (splitSize - 2);
        float splitsize = (float)XDIM - 2 / (float)splitSizeReal;/* top and bottom padding rows */
        
        /* Read in grid dimension dim and allocate a and b */
        printf("Using %d nodes.\n", numtasks);
        printf("Using [%d][%d] grid.\n", XDIM, YDIM);
        printf("Splitting in to arrays of size %d, + 2 padding\n", (splitSize - 2));

        /* Initialize grid*/
        printf("Initializing grid......\n");
        
        /* Add initial data */
        initdata(XDIM, YDIM, full_u_a);
        
        /* Output initial data */
        sprintf(buf,"initial_new_%d%s",1,str_end);
        prtdata(XDIM, YDIM, 0, full_u_a, buf);
        
        /*For each TimeStep*/
        for (it = 1; it <= params.nts; it++)
        {
            int numRows = splitSizeReal;
            int rowi = 0;/*Row iterator - jumps by number of rows - 2*/
            int nodei = 1;/*Node iterator*/

            //For each row-group
            while(rowi < XDIM - 2)
            {
                
                int endRow = rowi + numRows + 1; // end row
                
                //Stop from overflowing the end of the master matrix
                if(endRow >= YDIM)
                {
                    endRow = YDIM - 1;
                }
                
                /*****
                PERFORM SPLIT FUNCTION
                split full array into segments
                void split(double *fMat, double *split, int startRow, int endRow)
                *****/
                if(it&1)
                { /* it is odd */
                    split(full_u_a, broken_u_a, rowi, endRow);
                }
                else
                { /* it is even */
                    split(full_u_b, broken_u_a, rowi, endRow);
                }



                /*******
                SEND DATA OFF TO THE NODES
                Sending:
                    Int StartRow
                    Int EndRow
                    Double[] splitMatrix
                ********/
                MPI_Send(&rowi, 1, MPI_INT, nodei, 1, MPI_COMM_WORLD);
                MPI_Send(&endRow, 1, MPI_INT, nodei, 2, MPI_COMM_WORLD);

                int numRowsSent = endRow - rowi + 1;
                int numCellsSent = numRowsSent * XDIM;
                
                printf("> %d sending %d rows %d - %d\n", MASTER, nodei, rowi, endRow);

                /*
                ***printf("Sent value %8.3f\n", broken_u[5][5]);
                ***printf("numCellsSent from 0 %d\n", numCellsSent);
                */

                MPI_Send(broken_u_a, numCellsSent, MPI_DOUBLE, nodei, 3, MPI_COMM_WORLD);
                                
                rowi += numRows;
                nodei += 1;
            }

            /*
            Sometimes doesn't divide correctly.
            For example 100*100 on 12 cores.
            12 - 1(MASTER) = 11
            100-2 = 98
            98 / 11 = 8.9 (ceil) = 9
            11 * 9 = 99
            */
            if(nodei < numtasks){
                while(nodei < numtasks){
                    /* Send the node dummy data - will return 0 and be rejected */
                    rowi = 0;
                    int endRow = 0;
                    MPI_Send(&rowi, 1, MPI_INT, nodei, 1, MPI_COMM_WORLD);
                    MPI_Send(&endRow, 1, MPI_INT, nodei, 2, MPI_COMM_WORLD);
                    MPI_Send(broken_u_a, 0, MPI_DOUBLE, nodei, 3, MPI_COMM_WORLD);
                    nodei += 1;
                }
            }



            /****
            PERFORM COMBINE FUNCTION
            combine segment into the new full array
            *****/
            nodei = 1;/*Set back to 1 - Iterate over all nodes > MASTER again*/
            while (nodei < numtasks) {
                int startRowBack, endRowBack;
                
                MPI_Recv(&startRowBack, 1, MPI_INT, nodei, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(&endRowBack, 1, MPI_INT, nodei, 2, MPI_COMM_WORLD, &status);

                /*Calculate size of data returned*/
                int numRowBack = endRowBack - startRowBack + 1;/*Number of rows in full data set sent back*/
                int numCellBack = numRowBack * XDIM;

                MPI_Recv(broken_u_b, numCellBack, MPI_DOUBLE, nodei, 3, MPI_COMM_WORLD, &status);
                /*printf("< %d Node received from %d matrix rows from %d to %d\n",rank, nodei, startRowBack, endRowBack );*/
                /*printf("Received value %8.3f\n", broken_u_new[5][5]);*/
                
                if(it&1){
                    combine(broken_u_b, full_u_b, startRowBack, endRowBack);
                }else{
                    combine(broken_u_b, full_u_a, startRowBack, endRowBack);
                }
                nodei += 1;
            }

            /* Output the results for each timestep */
            sprintf(buf,"final_new_%d%s",it,str_end);
            if(it&1){
                prtdata(XDIM, YDIM, it-1, full_u_a, buf);
            } else {
                prtdata(XDIM, YDIM, it-1, full_u_b, buf);
            }

        }/* End for each timestep*/


        /***
        Output the final results
        ***/
        sprintf(buf,"final_new_%d%s",it,str_end);
        printf("Done. Created output file: %d\n", it);
        if(it&1){
            prtdata(XDIM, YDIM, it-1, full_u_a, buf);
        } else {
            prtdata(XDIM, YDIM, it-1, full_u_b, buf);
        }
        diff = clock() - start;
        int msec = diff;
        printf("Time taken %d milliseconds", msec);
    }
    else
    {
        /***********
        Run on other nodes (> MASTER)
        ***********/
        int startRowBack, endRowBack, numRowsSent, numCellsSent, ix, iy, it;

        for (it = 1; it <= params.nts; it++) {/*Iterate over timesteps on clients.*/
            MPI_Recv(&startRowBack, 1, MPI_INT, MASTER, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(&endRowBack, 1, MPI_INT, MASTER, 2, MPI_COMM_WORLD, &status);

            numRowsSent = endRowBack - startRowBack + 1;
            numCellsSent = numRowsSent * XDIM;

            MPI_Recv(broken_u_a, numCellsSent, MPI_DOUBLE, MASTER, 3, MPI_COMM_WORLD, &status);
            /*printf("Received value %8.3f\n", broken_u[5][5]);*/

            //printf("> %d Node received matrix rows from %d to %d\n",rank, startRowBack, endRowBack );
            /*printf("> %d Node received %d rows, %d cell\n",rank, numRowsSent, numCellsSent );*/
            
            /******PERFORM UPDATE FUNCTION*****/
            update(numRowsSent, YDIM, broken_u_a, broken_u_b);

            /*printf("Updated value %8.3f\n", broken_u_new[5][5]);*/

            /******SEND DATA BACK**********/
            /*printf("%d node sending matrix back to 0\n",rank); */
            /*Send our data back so MASTER knows where it fits in*/
            MPI_Send(&startRowBack, 1, MPI_INT, MASTER, 1, MPI_COMM_WORLD);
            MPI_Send(&endRowBack, 1, MPI_INT, MASTER, 2, MPI_COMM_WORLD);
            /*This could be refined by not sending back top and bottom rows - not needed.*/
            MPI_Send(broken_u_b, numCellsSent, MPI_DOUBLE, MASTER, 3, MPI_COMM_WORLD);
        }
    }
    
    MPI_Finalize();
    return 0;
}


/***
 * 
 Split the data in to a new matrix between row a and b.
 Updates splitMat using pointer
 **/
void split(double *fMat, double *splitMat, int startRow, int endRow)
{
    
    int numRows = endRow - startRow + 1;//Rows = number of rows between start and end + 1 above, 1 below
    int ix,iy;
    printf("Splitting rows %d to %d\n",startRow, endRow);
    //Copy data from old matrix to new matrix
    for (ix=0; ix<numRows; ix++)
    {
        int currentRow = startRow + ix;
        for (iy=0; iy<XDIM; iy++)
        {
            *(splitMat + (ix * XDIM) + iy) = *(fMat + (currentRow * XDIM) + iy);//Copy data from old array
        }
    }
}



/*
Recombine data from splitMatrix back in to target Matrix between start and end row
 */
void combine(double *segMatrix, double *tarMatrix, int startRow, int endRow)
{
    int numRows = endRow - startRow - 1;//We only want rows between the two numbers
    int i,j;
    //given row above and row below we want rows in between thus skip row 0 and stop before row-1
    for(i=1;i<=numRows;i++)
    {
        //due to boundaries skip col 0 and cols-1
        int currentRow = startRow + i;
        for(j=1;j<YDIM-1;j++)
        {
            // copy value from segmented matrix to the full matrix
            *(tarMatrix + ( currentRow*YDIM) + j) = *(segMatrix + (i * YDIM) + j);
        }
    }
}




/***
 * update: computes new values for timestep t+delta_t
 ***/
void update(int nx, int ny, double *u1, double *u2)
{
    int ix, iy;
    
    for (ix = 1; ix <= nx-2; ix++) {
        for (iy = 1; iy <= ny-2; iy++) {
            *(u2+ix*ny+iy) = *(u1+ix*ny+iy)  +
            params.cx * (*(u1+(ix+1)*ny+iy) + *(u1+(ix-1)*ny+iy) -
                         2.0 * *(u1+ix*ny+iy)) +
            params.cy * (*(u1+ix*ny+iy+1) + *(u1+ix*ny+iy-1) -
                         2.0 * *(u1+ix*ny+iy));
        }
    }
}



/***
 *  initdata: initializes array, timestep t=0
 * TODO: This function is returning negative values for large matrix sizes. > 1000*1000
 ***/
void initdata(int nx, int ny, double *u1)
{
    int ix, iy;
    for (ix = 0; ix <= nx-1; ix++)
        for (iy = 0; iy <= ny-1; iy++)
            *(u1+ix*ny+iy) = (double)(ix * (nx - ix - 1) * iy * (ny - iy - 1));
    
}




/***
 *  printdata: generates a .csv (well close enough) file with data contained in parameter double* u1
 ***/
void prtdata(int nx, int ny, int ts, double *u1, char* fname)
{
    int ix, iy;
    FILE *fp;
    char ffname[100];
    sprintf(ffname,"output/%s",fname);
    fp = fopen(ffname, "w");
    if(fp!=NULL)
    {
        for (ix = 0; ix < nx; ix++) {
            for (iy = 0; iy < ny; iy++) {
                fprintf(fp, "%8.3f,", *(u1+(ix*ny)+iy));
                if (iy != nx-1) {
                    fprintf(fp, " ");
                }
                else {
                    fprintf(fp, "\n");
                }
            }
        }
    }
    fclose(fp);
    printf(" %s\n",ffname);
}

