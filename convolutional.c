#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

// ran1 parameters
#define IA 16807
#define IM 2147483647
#define AM (1.0 / IM)
#define IQ 127773
#define IR 2836
#define NTAB 32
#define NDIV (1 + (IM - 1) / NTAB)
#define EPS 1.2e-7
#define RNMX (1.0 - EPS)

typedef struct
{
    int state;
    int input;
    int next_state;
    int output;
} StateRow;

typedef struct
{
    int data1;
    int data2;
    double data3;
    long data4;
    int data5;
} Data;

Data readFile(const char *);
void writeToFile_hard(const char *, int *, int *, int, int);
void writeToFile_soft(const char *, double *, int *, int, int);
//==================================================================
int next_state(int, int, int);
int calc_output(int, int, int, int);
int count_bits_set_to_one(int, int);
StateRow *conv_code_state_table_generator(int, int, int);
void state_table_printf(StateRow *, int);
int **state_transition_table_generator(StateRow *, int);
int **output_table_generator(StateRow *, int);
int *generated_sequence(int, int);
void Encoder(int **, int **, int *, int, int *);
double modulator(int);
//==================================================================
double noise_variance(double);
long *rand_init(long *, long);
double ran1(long *);
void normal(long *, double *, double);
//==================================================================
int **branch_table_generator(int **, int **, int);
int **survivor_init(int, int);
int **metric_hard_init(int);
double **metric_soft_init(int);
//==================================================================
void BER_compute(int *, int *, int);
int hard_decision(double);
void free2DArray(int **, int);
int calculate_state_weight(int, int);
//==================================================================
void decoder_hard(int **, int **, int **, int **, int *, int *, int *, int, int, int, int);
void survivor_metric_unit_hard(int **, int **, int, int, int, int);
void ACS_unit_hard(int **, int **, int *, int *, int *, int *, int, int, int);
void decode_output_hard(int **, int **, int **, int *, int, int, int, int);
void remaining_decode_output_hard(int **, int **, int *, int, int, int);
//==================================================================
void decoder_soft(int **, int **, int **, double **, int *, int *, double *, int, int, int, int);
void survivor_metric_unit_soft(double **, int **, int, double, int, int);
void ACS_unit_soft(double **, int **, int *, double *, double *, double *, int, int, int);
void decode_output_soft(int **, double **, int **, int *, int, int, int, int);
void remaining_decode_output_soft(int **, int **, int *, int, int, int);

int main()
{

    // int g1 = 0b11101;
    // int g2 = 0b01111;
    // int constraint_length = 5;

    int g1 = 0b011011;         // Generator polynomial 1: x^2 + x^3 + x^5 + x^6
    int g2 = 0b111001;         // Generator polynomial 2: x + x^2 + x^3 + x^6
    int constraint_length = 6; // The constraint length for the convolutional encoder

    // int g1 = 0b11101011;
    // int g2 = 0b01110001;
    // int constraint_length = 8;

    // Read the simulation parameters from the input file
    Data input_sim = readFile("Sim.txt");
    int truncation_length = input_sim.data1;
    int truncation_window_length = input_sim.data2;
    double SNR = input_sim.data3;
    long seed = input_sim.data4;
    int decision = input_sim.data5;

    // Generate the convolutional code state table using generator polynomials and constraint length
    StateRow *state_table = conv_code_state_table_generator(g1, g2, constraint_length);
    // state_table_printf(state_table,constraint_length);

    // Create state transition, output, and ACS unit tables
    int **state_transition_table = state_transition_table_generator(state_table, constraint_length);
    int **output_table = output_table_generator(state_table, constraint_length);
    int **branch_table = branch_table_generator(state_transition_table, output_table, constraint_length);

    // Initialize the survivor path
    int **survivor = survivor_init(truncation_window_length, constraint_length);

    // Initialize the random number generator
    long *idum = (long *)calloc(1, sizeof(long));
    rand_init(idum, seed);

    // Compute the standard deviation of the noise
    double variance = noise_variance(SNR);
    double sigma = sqrt(variance);

    // Generate a sequence of data to be transmitted
    int *u_tx = generated_sequence(truncation_length, constraint_length);
    int *u_rx = (int *)calloc((truncation_length + constraint_length), sizeof(int));

    // Initialize variables to keep track of the current state and the end of the survivor path
    int *current_state = (int *)calloc(1, sizeof(int));
    int *rear = (int *)calloc(1, sizeof(int));
    *rear = -1;
    int counter = 0;

    // Start the timer to measure the execution time of the simulation
    clock_t start, end;
    double cpu_time_used;
    start = clock();

    // If decision is 0, perform hard-decision decoding, otherwise perform soft-decision decoding
    if (decision == 0)
    {
        // Initialize the metrics table for hard decision
        int **metric_hard = metric_hard_init(constraint_length);

        // Allocate space for the hard decision output
        int *hard_decision_output = (int *)calloc((2 * (truncation_length + constraint_length)), sizeof(int));

        // Traverse through all the bits in the sequence
        for (int i = 0; i < truncation_length + constraint_length; ++i)
        {
            // Fetch the current bit from the sequence
            int u = u_tx[i];

            // Encode the current bit
            int c[2] = {0};
            Encoder(state_transition_table, output_table, current_state, u, c);

            // Initialize the array to hold the modulated encoded bits
            double x[2] = {0.0};
            x[0] = modulator(c[0]);
            x[1] = modulator(c[1]);

            // Add AWGN to the modulated signal
            double y[2] = {0.0}, z[2] = {0.0};
            normal(idum, z, sigma);
            y[0] = x[0] + z[0];
            y[1] = x[1] + z[1];

            // Perform hard decision decoding
            int y_hard[2] = {0};
            y_hard[0] = hard_decision(y[0]);
            y_hard[1] = hard_decision(y[1]);
            hard_decision_output[2 * i] = y_hard[0];
            hard_decision_output[2 * i + 1] = y_hard[1];

            // Use the Viterbi algorithm for hard-decision decoding
            decoder_hard(state_transition_table, branch_table, survivor, metric_hard,
                         rear, u_rx, y_hard, counter, constraint_length, truncation_window_length, truncation_length);

            ++counter;
        }

        // Perform the remaining hard-decision decoding
        remaining_decode_output_hard(state_transition_table, survivor, u_rx,
                                     *rear, truncation_window_length, counter);

        // Write the hard decision output to a file
        writeToFile_hard("Output_Hard.txt", hard_decision_output, u_rx, truncation_length, constraint_length);
        free(hard_decision_output);
    }
    else
    {
        // Initialize the metrics tables
        double **metric_soft = metric_soft_init(constraint_length);

        // Allocate space for the hard decision output
        double *soft_decision_output = (double *)calloc((2 * (truncation_length + constraint_length)), sizeof(double));

        // Traverse through all the bits in the sequence
        for (int i = 0; i < truncation_length + constraint_length; ++i)
        {
            // Fetch the current bit from the sequence
            int u = u_tx[i];

            // Encode the bit
            int c[2] = {0};
            Encoder(state_transition_table, output_table, current_state, u, c);

            // Initialize the array to hold the modulated encoded bits
            double x[2] = {0.0};
            x[0] = modulator(c[0]);
            x[1] = modulator(c[1]);

            // Add AWGN to the modulated signal
            double y[2] = {0.0}, z[2] = {0.0};
            normal(idum, z, sigma);
            y[0] = x[0] + z[0];
            y[1] = x[1] + z[1];

            // Perform soft decision decoding
            soft_decision_output[2 * i] = y[0];
            soft_decision_output[2 * i + 1] = y[1];

            // Use the Viterbi algorithm for soft-decision decoding
            decoder_soft(state_transition_table, branch_table, survivor, metric_soft,
                         rear, u_rx, y, counter, constraint_length, truncation_window_length, truncation_length);

            ++counter;
        }
        // Perform the remaining soft-decision decoding
        remaining_decode_output_soft(state_transition_table, survivor, u_rx,
                                     *rear, truncation_window_length, counter);
        // Write the soft decision output to a file
        writeToFile_soft("Output_Soft.txt", soft_decision_output, u_rx, truncation_length, constraint_length);
        free(soft_decision_output);
    }

    // Compute and print the bit error rate
    BER_compute(u_tx, u_rx, truncation_length);

    // Free all dynamically allocated memory
    free(idum);
    free(u_tx);
    free(u_rx);
    free(current_state);
    free(rear);

    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("running time: %f sec\n", cpu_time_used);
    system("pause");
    return 0;
}

// Function to free the memory allocated for a 2D array
void free2DArray(int **array, int rows)
{
    for (int i = 0; i < rows; i++)
    {
        free(array[i]);
    }
    free(array);
}

// Calculate the next state based on the current state, input, and constraint length
int next_state(int state, int input, int constraint_length)
{
    int reg = (state >> 1) | (input << (constraint_length - 1));
    return reg;
}

// Calculate the output for a given state, input, and generator polynomial
int calc_output(int state, int input, int gen_poly, int constraint_length)
{
    return (input + count_bits_set_to_one(state & gen_poly, constraint_length)) % 2;
}

// Count the number of bits set to one in the given number with specified bit length
int count_bits_set_to_one(int number, int bit_length)
{
    int count = 0;
    unsigned int mask = 1;

    for (int i = 0; i < bit_length; ++i)
    {
        if (number & mask)
        {
            ++count;
        }
        mask <<= 1;
    }

    return count;
}

// Generate a state table for the convolutional code based on generator polynomials and constraint length
StateRow *conv_code_state_table_generator(int g1, int g2, int constraint_length)
{
    int state_count = 1 << (constraint_length);
    StateRow *state_table = (StateRow *)malloc(state_count * 2 * sizeof(StateRow));
    int index = 0;

    for (int state = 0; state < state_count; ++state)
    {
        for (int input = 0; input < 2; ++input)
        {
            int next_state_val = next_state(state, input, constraint_length);
            int output1 = calc_output(state, input, g1, constraint_length);
            int output2 = calc_output(state, input, g2, constraint_length);

            StateRow row = {state, input, next_state_val, (output1 << 1) + output2};
            state_table[index++] = row;
        }
    }

    return state_table;
}

// Print the state table for debugging purposes
void state_table_printf(StateRow *state_table, int constraint_length)
{
    int state_count = 1 << (constraint_length);

    for (int i = 0; i < state_count * 2; ++i)
    {
        printf("State: %02d Input: %d Next state: %02d Output: %d%d\n",
               state_table[i].state, state_table[i].input, state_table[i].next_state,
               (state_table[i].output >> 1), state_table[i].output & 0b01);
        fflush(stdout);
    }
}

// Generate a state transition table from the state table for the convolutional code
int **state_transition_table_generator(StateRow *state_table, int constraint_length)
{
    int state_count = 1 << (constraint_length);
    int **state_transition_table = (int **)malloc(state_count * sizeof(int *));
    for (int i = 0; i < state_count; i++)
    {
        state_transition_table[i] = (int *)malloc(2 * sizeof(int));
        state_transition_table[i][0] = state_table[2 * i].next_state;
        state_transition_table[i][1] = state_table[2 * i + 1].next_state;
    }

    return state_transition_table;
}

// Generate an output table from the state table for the convolutional code
int **output_table_generator(StateRow *state_table, int constraint_length)
{
    int state_count = 1 << (constraint_length);
    int **output_table = (int **)malloc(state_count * sizeof(int *));
    for (int i = 0; i < state_count; i++)
    {
        output_table[i] = (int *)malloc(2 * sizeof(int));
        output_table[i][0] = state_table[2 * i].output;
        output_table[i][1] = state_table[2 * i + 1].output;
    }

    return output_table;
}

// Generate a sequence of length 'truncation_length + constraint_length' using a specified rule
int *generated_sequence(int truncation_length, int constraint_length)
{
    int *u_tx = (int *)malloc((truncation_length + constraint_length) * sizeof(int));
    // Set the first element to 1
    u_tx[0] = 1;
    for (int i = 1; i < (truncation_length + constraint_length); i++)
        u_tx[i] = 0;

    // Apply the specified rule to generate the sequence starting from the 7th element
    for (int i = 6; i < truncation_length; i++)
        u_tx[i] = (u_tx[i - 6] + u_tx[i - 5]) % 2;

    // Return the generated sequence
    return u_tx;
}

// Function to read a file and store data into a Data structure
Data readFile(const char *filename)
{
    FILE *file;
    char line[256];
    Data result = {0};

    file = fopen(filename, "r");
    if (file == NULL)
    {
        printf("無法打開文件\n");
        exit(1);
    }

    // Initialize a count to track the current line number
    int count = 0;
    // Read lines from the file until the maximum line number is reached
    while (fgets(line, sizeof(line), file) && count < 5)
    {
        // Check if the line contains a comment (indicated by '%')
        char *comment = strchr(line, '%');
        // Check if the line contains a comment (indicated by '%')
        if (comment)
        {
            *comment = '\0';
        }
        // Process the line differently based on the line number
        if (count == 2)
        {
            sscanf(line, "%lf", &result.data3);
        }
        else if (count == 3)
        {
            sscanf(line, "%ld", &result.data4);
        }
        else
        {
            int value;
            if (sscanf(line, "%d", &value) == 1)
            {
                switch (count)
                {
                case 0:
                    result.data1 = value;
                    break;
                case 1:
                    result.data2 = value;
                    break;
                case 4:
                    result.data5 = value;
                    break;
                default:
                    break;
                }
            }
        }
        count++;
    }
    fclose(file);

    return result;
}

// Encodes the input using the provided state transition table and output table, and updates the current state
void Encoder(int **state_transition_table, int **output_table, int *current_state, int input, int *c)
{
    int output = output_table[*current_state][input];
    *current_state = state_transition_table[*current_state][input];
    c[0] = output >> 1;
    c[1] = output & 0b01;
}

// Modulates the input value 'c' to 1 or -1, based on the input value
inline double modulator(int c)
{
    return (c == 1) ? -1.0 : 1.0;
}

// Function to write hard decision output and decoded information bit to a file
void writeToFile_hard(const char *filename, int *hard_decision_output, int *u_rx, int truncation_length, int constraint_length)
{
    {
        FILE *file = fopen(filename, "w");
        if (file == NULL)
        {
            printf("無法打開文件\n");
            return;
        }

        // Write the hard decision output to the file
        for (int i = 0; i < 2 * (truncation_length + constraint_length); i++)
        {
            fprintf(file, "%d ", hard_decision_output[i]);
        }
        fprintf(file, " hard-decision output: 2*(N+m) elements\n");

        // Write the decoded information bit to the file
        for (int i = 0; i < (truncation_length + constraint_length); i++)
        {
            fprintf(file, "%d ", u_rx[i]);
        }

        fprintf(file, " decoded information bit\n");
        fclose(file);
    }
}

// Function to write soft decision output and decoded information bit to a file
void writeToFile_soft(const char *filename, double *soft_decision_output, int *u_rx, int truncation_length, int constraint_length)
{
    {
        FILE *file = fopen(filename, "w");
        if (file == NULL)
        {
            printf("無法打開文件\n");
            return;
        }
        // Write the soft decision output to the file
        for (int i = 0; i < 2 * (truncation_length + constraint_length); i++)
        {
            fprintf(file, "%.6f ", soft_decision_output[i]);
        }
        fprintf(file, " y: 2*(N+m) elements\n");

        // Write the decoded information bit to the file
        for (int i = 0; i < (truncation_length + constraint_length); i++)
        {
            fprintf(file, "%d ", u_rx[i]);
        }

        fprintf(file, " decoded information bit\n");
        fclose(file);
    }
}

// Function to calculate noise variance given the Signal-to-Noise Ratio (SNR)
double noise_variance(double SNR)
{
    return 1 / pow(10, SNR / 10);
}

// Function to initialize the random number generator with a given seed
long *rand_init(long *idum, long seed)
{
    *idum = seed;
    return idum;
}

// Function to generate a uniform random number in the range (0, 1) using a linear congruential generator
double ran1(long *idum)
{
    int j;
    long k;
    static long iy = 0;
    static long iv[NTAB];
    double temp;

    if (*idum <= 0 || !iy)
    {
        if (-(*idum) < 1)
            *idum = 1;
        else
            *idum = -(*idum);
        for (j = NTAB + 7; j >= 0; j--)
        {
            k = (*idum) / IQ;
            *idum = IA * (*idum - k * IQ) - IR * k;
            if (*idum < 0)
                *idum += IM;
            if (j < NTAB)
                iv[j] = *idum;
        }
        iy = iv[0];
    }
    k = (*idum) / IQ;
    *idum = IA * (*idum - k * IQ) - IR * k;
    if (*idum < 0)
        *idum += IM;
    j = iy / NDIV;
    iy = iv[j];
    iv[j] = *idum;

    // Normalize and return the random number
    if ((temp = AM * iy) > RNMX)
        return RNMX;
    else
        return temp;
}

// Function to generate two normally distributed random numbers using the Box-Muller transform
void normal(long *idum, double *n, double sigma)
{
    double x1, x2, s;

    // Generate two independent uniform random variables in the range (-1, 1)
    do
    {
        x1 = ran1(idum);
        x2 = ran1(idum);
        x1 = 2.0 * x1 - 1.0;
        x2 = 2.0 * x2 - 1.0;
        s = pow(x1, 2) + pow(x2, 2); // s is the square of the distance from the origin
    } while (s >= 1.0);              // Repeat if the point is outside the unit circle

    // Use the Box-Muller transform to generate two normally distributed numbers
    n[0] = sigma * x1 * sqrt((-2) * log(s) / s);
    n[1] = sigma * x2 * sqrt((-2) * log(s) / s);
}

// Function to generate the Add-Compare-Select (ACS) unit table for Viterbi decoding
int **branch_table_generator(int **state_transition_table, int **output_table, int constraint_length)
{
    // Calculate the number of states from the constraint length
    int state_count = 1 << (constraint_length);

    // Allocate memory for the ACS unit table
    int **branch_table = (int **)malloc(state_count * sizeof(int *));
    for (int i = 0; i < state_count; i++)
    {
        // Each state has two incoming branches, each with a predecessor state and two encoded bits
        branch_table[i] = (int *)malloc(6 * sizeof(int));
        int counter = 0;

        // Find the two predecessor states and the associated encoded bits for each state
        for (int j = 0; j < state_count; j++)
        {
            // If the current state is reached by the 0-branch from state j
            if (state_transition_table[j][0] == i)
            {
                if (counter == 0)
                {
                    // Save the state j and the corresponding outputs
                    branch_table[i][0] = j;
                    branch_table[i][1] = output_table[j][0] >> 1;
                    branch_table[i][2] = output_table[j][0] & 0b01;
                    counter++;
                }

                else
                {
                    // Save the second state j and the corresponding outputs
                    branch_table[i][3] = j;
                    branch_table[i][4] = output_table[j][0] >> 1;
                    branch_table[i][5] = output_table[j][0] & 0b01;
                    break;
                }
            }
            // If the current state is reached by the 1-branch from state j
            else if (state_transition_table[j][1] == i)
            {
                if (counter == 0)
                {
                    // Save the state j and the corresponding outputs
                    branch_table[i][0] = j;
                    branch_table[i][1] = output_table[j][1] >> 1;
                    branch_table[i][2] = output_table[j][1] & 0b01;
                    counter++;
                }
                else
                {
                    // Save the second state j and the corresponding outputs
                    branch_table[i][3] = j;
                    branch_table[i][4] = output_table[j][1] >> 1;
                    branch_table[i][5] = output_table[j][1] & 0b01;
                    break;
                }
            }
        }
    }
    return branch_table;
}

// Function to initialize the survivor path 2D array for Viterbi algorithm
int **survivor_init(int truncation_window_length, int constraint_length)
{
    int state_count = 1 << (constraint_length);
    int **survivor = (int **)malloc(state_count * sizeof(int *));
    for (int i = 0; i < state_count; i++)
    {
        survivor[i] = (int *)calloc(truncation_window_length, sizeof(int));
    }
    return survivor;
}

// Function to initialize the hard decision metric 2D array for Viterbi algorithm
int **metric_hard_init(int constraint_length)
{
    int state_count = 1 << (constraint_length);
    int **metric_hard = (int **)malloc(state_count * sizeof(int *));
    for (int i = 0; i < state_count; i++)
    {
        metric_hard[i] = (int *)malloc(2 * sizeof(int));
        // Initialize each metric with a large value (INT_MAX)
        for (int j = 0; j < 2; j++)
            metric_hard[i][j] = INT_MAX;
    }
    // Set the metric of the initial state to 0
    metric_hard[0][0] = 0;
    return metric_hard;
}

// Function to initialize the soft decision metric 2D array for Viterbi algorithm
double **metric_soft_init(int constraint_length)
{
    int state_count = 1 << (constraint_length);
    double **metric_soft = (double **)malloc(state_count * sizeof(double *));
    for (int i = 0; i < state_count; i++)
    {
        metric_soft[i] = (double *)malloc(2 * sizeof(double));
        // Initialize each metric with a large value (DBL_MAX)
        for (int j = 0; j < 2; j++)
            metric_soft[i][j] = DBL_MAX;
    }
    // Set the metric of the initial state to 0.0
    metric_soft[0][0] = 0.0;
    return metric_soft;
}

// Hard decision decoder function for Convolutional codes
void decoder_hard(int **state_transition_table, int **branch_table, int **survivor, int **metric_hard,
                  int *rear, int *u_rx, int *y_hard, int counter, int constraint_length, int truncation_window_length, int truncation_length)
{
    int state_count = 1 << (constraint_length);

    // Call the hard output decoding function once the counter exceeds or equals truncation_window_length
    if (counter >= truncation_window_length)
    {
        decode_output_hard(state_transition_table, metric_hard, survivor,
                           u_rx, *rear, constraint_length, truncation_window_length, counter);
    }

    // Update the rear index by incrementing it and wrapping around the truncation window length
    *rear = (*rear + 1) % truncation_window_length;

    // Loop through each state to compute the metrics
    for (int index = 0; index < state_count; index++)
    {
        // Get the previous states associated with the current state from the ACS unit table
        int prevstate[2] = {0};
        prevstate[0] = branch_table[index][0];
        prevstate[1] = branch_table[index][3];

        // Get the hard decision values associated with the current state from the ACS unit table
        int c_hard[4] = {0};
        c_hard[0] = branch_table[index][1];
        c_hard[1] = branch_table[index][2];
        c_hard[2] = branch_table[index][4];
        c_hard[3] = branch_table[index][5];


        if (counter < truncation_length)
        {   

            // Initialize the new metric hard array
            int new_metric_hard[2] = {0};

            // Compute the Add-Compare-Select (ACS) unit for the current state
            ACS_unit_hard(metric_hard, survivor, prevstate, new_metric_hard, c_hard, y_hard, constraint_length, *rear, index);
        }

        //zero-tail,set the state back to 0
        else
        {
            //Determine whether the decoding information bit is 0, if so, it is a valid survivor
            if (state_transition_table[prevstate[0]][0] == index && state_transition_table[prevstate[1]][0] == index)
            {

                // Initialize the new metric hard array
                int new_metric_hard[2] = {0};

                // Compute the Add-Compare-Select (ACS) unit for the current state
                ACS_unit_hard(metric_hard, survivor, prevstate, new_metric_hard, c_hard, y_hard, constraint_length, *rear, index);
            }
            else
            {
                survivor_metric_unit_hard(metric_hard, survivor, -1, INT_MAX, *rear, index);
            }
        }
    }

    // Move the temporary metric values to the 0's column for the next iteration
    for (int i = 0; i < state_count; i++)
    {
        metric_hard[i][0] = metric_hard[i][1];
    }
}

// Soft decision decoder function for Convolutional codes
void decoder_soft(int **state_transition_table, int **branch_table, int **survivor, double **metric_soft,
                  int *rear, int *u_rx, double *y, int counter, int constraint_length, int truncation_window_length, int truncation_length)
{
    int state_count = 1 << (constraint_length);

    // decode_output_soft
    if (counter >= truncation_window_length)
    {
        decode_output_soft(state_transition_table, metric_soft, survivor,
                           u_rx, *rear, constraint_length, truncation_window_length, counter);
    }

    //
    *rear = (*rear + 1) % truncation_window_length;
    for (int index = 0; index < state_count; index++)
    {
        int prevstate[2] = {0};
        prevstate[0] = branch_table[index][0];
        prevstate[1] = branch_table[index][3];

        // moudulate the referance signal
        double c_soft[4] = {0};
        c_soft[0] = modulator(branch_table[index][1]);
        c_soft[1] = modulator(branch_table[index][2]);
        c_soft[2] = modulator(branch_table[index][4]);
        c_soft[3] = modulator(branch_table[index][5]);

        if (counter < truncation_length)
        {
            // Initialize the new metric soft array
            double new_metric_soft[2] = {0.0};

            // Compute the Add-Compare-Select (ACS) unit for the current state
            ACS_unit_soft(metric_soft, survivor, prevstate, new_metric_soft, c_soft, y, constraint_length, *rear, index);
        }

        else
        {

            //
            if (state_transition_table[prevstate[0]][0] == index && state_transition_table[prevstate[1]][0] == index)
            {

                // Initialize the new metric soft array
                double new_metric_soft[2] = {0.0};

                // Compute the Add-Compare-Select (ACS) unit for the current state
                ACS_unit_soft(metric_soft, survivor, prevstate, new_metric_soft, c_soft, y, constraint_length, *rear, index);
            }
            else
            {
                survivor_metric_unit_soft(metric_soft, survivor, -1, DBL_MAX, *rear, index);
            }
        }
    }

    for (int i = 0; i < state_count; i++)
    {
        metric_soft[i][0] = metric_soft[i][1];
    }
}

// Function to compute the weight of a given state
int calculate_state_weight(int state, int constraint_length)
{
    int state_weight = 0;

    // Loop through each bit in the state's binary representation
    for (int i = 0; i < constraint_length; i++)
    {
        // Shift the state to the right by 'i' places, bitwise AND with 0b01 to isolate the bit at position 'i',
        // multiply it by 2 raised to the power of 'constraint_length - 1 - i', and add it to the state weight
        state_weight += ((state >> i) & 0b01) * pow(2, constraint_length - 1 - i);
    }
    return state_weight;
}

// Function to perform the Add-Compare-Select (ACS) operation in Viterbi algorithm for hard decisions
void ACS_unit_hard(int **metric_hard, int **survivor, int *prevstate, int *new_metric_hard,
                   int *c, int *y, int constraint_length, int rear, int index)
{
    // Compute the Hamming distance for the two possible transitions
    int hamDistance0 = (c[0] ^ y[0]) + (c[1] ^ y[1]);
    int hamDistance1 = (c[2] ^ y[0]) + (c[3] ^ y[1]);

    // Depending on the previous state metrics, compute the new metrics
    // or assign them a large value (INT_MAX)
    if (metric_hard[prevstate[0]][0] != INT_MAX && metric_hard[prevstate[1]][0] != INT_MAX)
    {
        new_metric_hard[0] = metric_hard[prevstate[0]][0] + hamDistance0;
        new_metric_hard[1] = metric_hard[prevstate[1]][0] + hamDistance1;
    }
    else if (metric_hard[prevstate[0]][0] == INT_MAX && metric_hard[prevstate[1]][0] != INT_MAX)
    {
        new_metric_hard[0] = INT_MAX;
        new_metric_hard[1] = metric_hard[prevstate[1]][0] + hamDistance1;
    }
    else if (metric_hard[prevstate[0]][0] != INT_MAX && metric_hard[prevstate[1]][0] == INT_MAX)
    {
        new_metric_hard[0] = metric_hard[prevstate[0]][0] + hamDistance0;
        new_metric_hard[1] = INT_MAX;
    }
    else
    {
        new_metric_hard[0] = INT_MAX;
        new_metric_hard[1] = INT_MAX;
    }

    // Compare the new metrics and decide which path to follow based on their value
    if (new_metric_hard[0] != INT_MAX && new_metric_hard[1] != INT_MAX)
    {
        // If metrics are not the same, choose the path with the smaller metric
        if (new_metric_hard[0] != new_metric_hard[1])
        {
            if (new_metric_hard[0] < new_metric_hard[1])
            {
                survivor_metric_unit_hard(metric_hard, survivor, prevstate[0], new_metric_hard[0], rear, index);
            }
            else
            {
                survivor_metric_unit_hard(metric_hard, survivor, prevstate[1], new_metric_hard[1], rear, index);
            }
        }
        // If metrics are the same, choose the path based on the state weight
        else
        {
            int prevState0_weight = calculate_state_weight(prevstate[0], constraint_length);
            int prevState1_weight = calculate_state_weight(prevstate[1], constraint_length);

            if (prevState0_weight > prevState1_weight)
            {
                survivor_metric_unit_hard(metric_hard, survivor, prevstate[1], new_metric_hard[1], rear, index);
            }
            else
            {
                survivor_metric_unit_hard(metric_hard, survivor, prevstate[0], new_metric_hard[0], rear, index);
            }
        }
    }
    // If only one metric is valid, choose that path
    else if (new_metric_hard[0] == INT_MAX && new_metric_hard[1] != INT_MAX)
    {
        survivor_metric_unit_hard(metric_hard, survivor, prevstate[1], new_metric_hard[1], rear, index);
    }
    else if (new_metric_hard[0] != INT_MAX && new_metric_hard[1] == INT_MAX)
    {
        survivor_metric_unit_hard(metric_hard, survivor, prevstate[0], new_metric_hard[0], rear, index);
    }
    // If neither metrics are valid, assign a large value
    else
    {
        survivor_metric_unit_hard(metric_hard, survivor, -1, INT_MAX, rear, index);
    }
}

// Function to perform the Add-Compare-Select (ACS) operation in Viterbi algorithm for soft decisions
void ACS_unit_soft(double **metric_soft, int **survivor, int *prevstate, double *new_metric_soft,
                   double *c, double *y, int constraint_length, int rear, int index)
{

    // Compute the Euclidean distance for the two possible transitions
    double euclidean_distance0 = pow((c[0] - y[0]), 2) + pow((c[1] - y[1]), 2);
    double euclidean_distance1 = pow((c[2] - y[0]), 2) + pow((c[3] - y[1]), 2);

    // Depending on the previous state metrics, compute the new metrics
    // or assign them a large value (DBL_MAX)
    if (metric_soft[prevstate[0]][0] != DBL_MAX && metric_soft[prevstate[1]][0] != DBL_MAX)
    {
        new_metric_soft[0] = metric_soft[prevstate[0]][0] + euclidean_distance0;
        new_metric_soft[1] = metric_soft[prevstate[1]][0] + euclidean_distance1;
    }
    else if (metric_soft[prevstate[0]][0] == DBL_MAX && metric_soft[prevstate[1]][0] != DBL_MAX)
    {
        new_metric_soft[0] = DBL_MAX;
        new_metric_soft[1] = metric_soft[prevstate[1]][0] + euclidean_distance1;
    }
    else if (metric_soft[prevstate[0]][0] != DBL_MAX && metric_soft[prevstate[1]][0] == DBL_MAX)
    {
        new_metric_soft[0] = metric_soft[prevstate[0]][0] + euclidean_distance0;
        new_metric_soft[1] = DBL_MAX;
    }
    else
    {
        new_metric_soft[0] = DBL_MAX;
        new_metric_soft[1] = DBL_MAX;
    }

    // Similar to the hard decision ACS operation, but now the metrics are floating-point numbers
    // and the comparison is based on the smaller floating-point value
    if (new_metric_soft[0] != DBL_MAX && new_metric_soft[1] != DBL_MAX)
    {
        // If metrics are not the same, choose the path with the smaller metric
        if (new_metric_soft[0] != new_metric_soft[1])
        {
            if (new_metric_soft[0] < new_metric_soft[1])
            {
                survivor_metric_unit_soft(metric_soft, survivor, prevstate[0], new_metric_soft[0], rear, index);
            }
            else
            {
                survivor_metric_unit_soft(metric_soft, survivor, prevstate[1], new_metric_soft[1], rear, index);
            }
        }
        // If metrics are the same, choose the path based on the state weight
        else
        {
            int prevState0_weight = calculate_state_weight(prevstate[0], constraint_length);
            int prevState1_weight = calculate_state_weight(prevstate[1], constraint_length);

            if (prevState0_weight > prevState1_weight)
            {
                survivor_metric_unit_soft(metric_soft, survivor, prevstate[1], new_metric_soft[1], rear, index);
            }
            else
            {
                survivor_metric_unit_soft(metric_soft, survivor, prevstate[0], new_metric_soft[0], rear, index);
            }
        }
    }
    // If only one metric is valid, choose that path
    else if (new_metric_soft[0] == DBL_MAX && new_metric_soft[1] != DBL_MAX)
    {
        survivor_metric_unit_soft(metric_soft, survivor, prevstate[1], new_metric_soft[1], rear, index);
    }
    else if (new_metric_soft[0] != DBL_MAX && new_metric_soft[1] == DBL_MAX)
    {
        survivor_metric_unit_soft(metric_soft, survivor, prevstate[0], new_metric_soft[0], rear, index);
    }
    // if neither metrics are valid, assign a large value (DBL_MAX)
    else
    {
        survivor_metric_unit_soft(metric_soft, survivor, -1, DBL_MAX, rear, index);
    }
}

// Function to update the survivor path and metric for the 'hard' decision case
inline void survivor_metric_unit_hard(int **metric_hard, int **survivor, int prevstate, int new_metric_hard, int rear, int index)
{
    survivor[index][rear] = prevstate;
    metric_hard[index][1] = new_metric_hard;
}

// Function to update the survivor path and metric for the 'soft' decision case
inline void survivor_metric_unit_soft(double **metric_soft, int **survivor, int prevstate, double new_metric_soft, int rear, int index)
{
    survivor[index][rear] = prevstate;
    metric_soft[index][1] = new_metric_soft;
}

// Function to decode the received message using a hard decision Viterbi decoder
void decode_output_hard(int **state_transition_table, int **metric_hard, int **survivor,
                        int *u_rx, int rear, int constraint_length, int truncation_window_length, int counter)
{
    int state_count = 1 << (constraint_length);

    // Initialize the best state as state 0 and the minimum metric as the metric of state 0
    int state = 0;
    int min = metric_hard[0][0];

    // Find the best state with the smallest metric in the current time step
    for (int i = 1; i < state_count; i++)
    {
        if (min != metric_hard[i][0])
        {
            if (min > metric_hard[i][0])
            {
                min = metric_hard[i][0];
                state = i;
            }
        }
        // If the metric of state i is equal to the current minimum metric
        else
        {
            int State0_weight = calculate_state_weight(state, constraint_length);
            int State1_weight = calculate_state_weight(i, constraint_length);
            if (State0_weight > State1_weight)
            {
                state = i;
            }
        }
    }

    // Trace back from the best state to find the most likely path
    int pre_state0 = 0, pre_state1 = 0;
    int index = rear;
    for (int i = 0; i < truncation_window_length; i++)
    {
        // Get the previous state
        pre_state0 = survivor[state][index];
        pre_state1 = state;

        // Update the current state to the previous state
        state = pre_state0;

        // Update the index for the survivor table
        index--;
        if (index == -1)
        {
            index = truncation_window_length - 1;
        }
    }

    // Decode the information bit based on the transition from the previous state to the current state
    if (state_transition_table[pre_state0][0] == pre_state1)
    {
        u_rx[counter - truncation_window_length] = 0;
    }
    else
    {
        u_rx[counter - truncation_window_length] = 1;
    }
}

// Function to decode the received message using a soft decision Viterbi decoder
void decode_output_soft(int **state_transition_table, double **metric_soft, int **survivor,
                        int *u_rx, int rear, int constraint_length, int truncation_window_length, int counter)
{
    int state_count = 1 << (constraint_length);

    // Initialize the best state as state 0 and the minimum metric as the metric of state 0
    int state = 0;
    double min = metric_soft[0][0];

    // Find the best state with the smallest metric in the current time step
    for (int i = 1; i < state_count; i++)
    {
        if (min != metric_soft[i][0])
        {
            if (min > metric_soft[i][0])
            {
                min = metric_soft[i][0];
                state = i;
            }
        }

        // If the metric of state i is equal to the current minimum metric
        else
        {
            int State0_weight = calculate_state_weight(state, constraint_length);
            int State1_weight = calculate_state_weight(i, constraint_length);
            if (State0_weight > State1_weight)
            {
                state = i;
            }
        }
    }

    // Trace back from the best state to find the most likely path
    int pre_state0 = 0, pre_state1 = 0;
    int index = rear;
    for (int i = 0; i < truncation_window_length; i++)
    {
        pre_state0 = survivor[state][index];
        pre_state1 = state;
        state = pre_state0;
        index--;
        if (index == -1)
        {
            index = truncation_window_length - 1;
        }
    }

    // decode information bit
    if (state_transition_table[pre_state0][0] == pre_state1)
    {
        u_rx[counter - truncation_window_length] = 0;
    }
    else
    {
        u_rx[counter - truncation_window_length] = 1;
    }
}

// Function to decode the remaining output using a hard decision Viterbi decoder
void remaining_decode_output_hard(int **state_transition_table, int **survivor,
                                  int *u_rx, int rear, int truncation_window_length, int counter)
{
    int state = 0;

    // Trace back from the best state to find the most likely path and decode the remaining bits
    int pre_state0 = 0, pre_state1 = 0;
    int index = rear;

    // Trace back for every bit in the truncation window
    for (int i = 0; i < truncation_window_length; i++)
    {
        // Trace back from the best state to find the most likely path
        for (int j = i; j < truncation_window_length; j++)
        {
            // Get the previous state
            pre_state0 = survivor[state][index];
            pre_state1 = state;

            // Update the current state to the previous state
            state = pre_state0;

            // Update the index for the survivor table
            index--;
            if (index == -1)
            {
                index = truncation_window_length - 1;
            }
        }

        // Decode the information bit based on the transition from the previous state to the current state
        if (state_transition_table[pre_state0][0] == pre_state1)
        {
            u_rx[counter - truncation_window_length] = 0;
        }
        else
        {
            u_rx[counter - truncation_window_length] = 1;
        }

        // Reset the state to the best state and the index to the rear for the next bit
        state = 0;
        index = rear;

        // Increase the counter for the next bit
        ++counter;
    }
}

// Function to decode the remaining output using a soft decision Viterbi decoder
void remaining_decode_output_soft(int **state_transition_table, int **survivor,
                                  int *u_rx, int rear, int truncation_window_length, int counter)
{
    int state = 0;

    // Trace back from the best state to find the most likely path and decode the remaining bits
    int pre_state0 = 0, pre_state1 = 0;
    int index = rear;

    // Trace back for every bit in the truncation window
    for (int i = 0; i < truncation_window_length; i++)
    {
        // Trace back from the best state to find the most likely path
        for (int j = i; j < truncation_window_length; j++)
        {
            // Get the previous state
            pre_state0 = survivor[state][index];
            pre_state1 = state;

            // Update the current state to the previous state
            state = pre_state0;

            // Update the index for the survivor table
            index--;
            if (index == -1)
            {
                index = truncation_window_length - 1;
            }
        }

        // Decode the information bit based on the transition from the previous state to the current state
        if (state_transition_table[pre_state0][0] == pre_state1)
        {
            u_rx[counter - truncation_window_length] = 0;
        }
        else
        {
            u_rx[counter - truncation_window_length] = 1;
        }

        // Reset the state to the best state and the index to the rear for the next bit
        state = 0;
        index = rear;

        // Increase the counter for the next bit
        ++counter;
    }
}

// Function to make a hard decision based on the received symbol y
inline int hard_decision(double y)
{
    return (y > 0.0) ? 0 : 1;
}

// Function to compute Bit Error Rate (BER) between transmitted and received data
void BER_compute(int *u_tx, int *u_rx, int truncation_length)
{

    double length = 0.0, BER = 0.0;
    int err_num = 0, i = 0;
    while (err_num < 1000 && i < truncation_length)
    {
        if (u_tx[i] != u_rx[i])
        {
            ++err_num;
        }
        i++;
    }
    length = i;
    BER = err_num / length;
    printf("BER= %f,bit errors= %d, total number of transferred bits= %d\n", BER, err_num, i);
}