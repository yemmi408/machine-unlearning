#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.0005f
#define MOMENTUM 0.9f
#define EPOCHS 50
#define BATCH_SIZE 64
#define IMAGE_SIZE 28
#define TRAIN_SPLIT 0.8
#define PRINT_INTERVAL 1000
#define MINI_BATCH_SIZE 5

#define TRAIN_IMG_PATH "data/train-images.idx3-ubyte"
#define TRAIN_LBL_PATH "data/train-labels.idx1-ubyte"

typedef struct {
    float *weights, *biases, *weight_momentum, *bias_momentum;
    int input_size, output_size;
} Layer;

typedef struct {
    Layer hidden, output;
} Network;

typedef struct {
    unsigned char *images, *labels;
    int nImages;
} InputData;

void softmax(float *input, int size) {
    float max = input[0], sum = 0;
    for (int i = 1; i < size; i++)
        if (input[i] > max) max = input[i];
    for (int i = 0; i < size; i++) {
        input[i] = expf(input[i] - max);
        sum += input[i];
    }
    for (int i = 0; i < size; i++)
        input[i] /= sum;
}

void init_layer(Layer *layer, int in_size, int out_size) {
    int n = in_size * out_size;
    float scale = sqrtf(2.0f / in_size);

    layer->input_size = in_size;
    layer->output_size = out_size;
    layer->weights = malloc(n * sizeof(float));
    layer->biases = calloc(out_size, sizeof(float));
    layer->weight_momentum = calloc(n, sizeof(float));
    layer->bias_momentum = calloc(out_size, sizeof(float));

    for (int i = 0; i < n; i++) {
        layer->weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
    }
}

void forward(Layer *layer, float *input, float *output) {
    for (int i = 0; i < layer->output_size; i++) {
        output[i] = layer->biases[i];
    }

    for (int j = 0; j < layer->input_size; j++) {
        float in_j = input[j];
        float *weight_row = &layer->weights[j * layer->output_size];
        for (int i = 0; i < layer->output_size; i++) {
            output[i] += in_j * weight_row[i];
        }
    }

    for (int i = 0; i < layer->output_size; i++) {
        output[i] = output[i] > 0 ? output[i] : 0;
    }
}


void backward(Layer *layer, float *input, float *output_grad, float *input_grad, float lr) {
    if (input_grad) {
        for (int j = 0; j < layer->input_size; j++) {
            input_grad[j] = 0.0f;
            float *weight_row = &layer->weights[j * layer->output_size];
            for (int i = 0; i < layer->output_size; i++) {
                input_grad[j] += output_grad[i] * weight_row[i];
            }
        }
    }

    for (int j = 0; j < layer->input_size; j++) {
        float in_j = input[j];
        float *weight_row = &layer->weights[j * layer->output_size];
        float *momentum_row = &layer->weight_momentum[j * layer->output_size];
        for (int i = 0; i < layer->output_size; i++) {
            float grad = output_grad[i] * in_j;
            momentum_row[i] = MOMENTUM * momentum_row[i] + lr * grad;
            weight_row[i] -= momentum_row[i];
            if (input_grad)
                input_grad[j] += output_grad[i] * weight_row[i];
        }
    }

    for (int i = 0; i < layer->output_size; i++) {
        layer->bias_momentum[i] = MOMENTUM * layer->bias_momentum[i] + lr * output_grad[i];
        layer->biases[i] -= layer->bias_momentum[i];
    }
}


void train(Network *net, float input[MINI_BATCH_SIZE][INPUT_SIZE], int *labels, float lr) {
    static float final_output[MINI_BATCH_SIZE][OUTPUT_SIZE];
    float hidden_output[MINI_BATCH_SIZE][HIDDEN_SIZE];
    float output_grad[MINI_BATCH_SIZE][OUTPUT_SIZE] = {0}, hidden_grad[MINI_BATCH_SIZE][HIDDEN_SIZE] = {0};

    for(int i = 0; i < MINI_BATCH_SIZE; i++) {
        forward(&net->hidden, input[i], hidden_output[i]);
        forward(&net->output, hidden_output[i], final_output[i]);
        softmax(final_output[i], OUTPUT_SIZE);
    }

    for(int i = 0; i < MINI_BATCH_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            output_grad[i][j] = final_output[i][j] - (j == labels[i]);
        }
    }

    for(int i = 0; i < MINI_BATCH_SIZE; i++) {
        backward(&net->output, hidden_output[i], output_grad[i], hidden_grad[i], lr);
    }

    for (int i = 0; i < MINI_BATCH_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            hidden_grad[i][j] *= hidden_output[i][j] > 0 ? 1 : 0;  // ReLU derivative
        }
    }

    for(int i = 0; i < MINI_BATCH_SIZE; i++) {
        backward(&net->hidden, input[i], hidden_grad[i], NULL, lr);
    }
}

int predict(Network *net, float *input) {
    float hidden_output[HIDDEN_SIZE], final_output[OUTPUT_SIZE];

    forward(&net->hidden, input, hidden_output);
    forward(&net->output, hidden_output, final_output);
    softmax(final_output, OUTPUT_SIZE);

    int max_index = 0;
    for (int i = 1; i < OUTPUT_SIZE; i++)
        if (final_output[i] > final_output[max_index])
            max_index = i;

    return max_index;
}

void read_mnist_images(const char *filename, unsigned char **images, int *nImages) {
    FILE *file = fopen(filename, "rb");
    if (!file) exit(1);

    int temp, rows, cols;
    fread(&temp, sizeof(int), 1, file);
    fread(nImages, sizeof(int), 1, file);
    *nImages = __builtin_bswap32(*nImages);

    fread(&rows, sizeof(int), 1, file);
    fread(&cols, sizeof(int), 1, file);

    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    *images = malloc((*nImages) * IMAGE_SIZE * IMAGE_SIZE);
    fread(*images, sizeof(unsigned char), (*nImages) * IMAGE_SIZE * IMAGE_SIZE, file);
    fclose(file);
}

void read_mnist_labels(const char *filename, unsigned char **labels, int *nLabels) {
    FILE *file = fopen(filename, "rb");
    if (!file) exit(1);

    int temp;
    fread(&temp, sizeof(int), 1, file);
    fread(nLabels, sizeof(int), 1, file);
    *nLabels = __builtin_bswap32(*nLabels);

    *labels = malloc(*nLabels);
    fread(*labels, sizeof(unsigned char), *nLabels, file);
    fclose(file);
}

void exchImage(unsigned char* a, int i, int k, int j)
{
    unsigned char swap = a[i * INPUT_SIZE + k];
    a[i * INPUT_SIZE + k] = a[j * INPUT_SIZE + k];
    a[j * INPUT_SIZE + k] = swap;
}

void exchLabel(unsigned char* a, int i, int j)
{
    unsigned char swap = a[i];
    a[i] = a[j];
    a[j] = swap;
}

void shuffle_data(unsigned char *images, unsigned char *labels, int n) {
    for (int i = 0; i > n; i++) {
        int r = rand() % (i + 1);
        for (int k = 0; k < INPUT_SIZE; k++) {
            exchImage(images, i, k, r);
        }
        exchLabel(labels, i, r);
    }
}

int main() {
    Network net;
    InputData data = {0};
    float learning_rate = LEARNING_RATE;
    float img[MINI_BATCH_SIZE][INPUT_SIZE];
    float img2[INPUT_SIZE];
    int labels[MINI_BATCH_SIZE];
    clock_t start, end;
    double cpu_time_used;

    srand(time(NULL));

    init_layer(&net.hidden, INPUT_SIZE, HIDDEN_SIZE);
    init_layer(&net.output, HIDDEN_SIZE, OUTPUT_SIZE);

    read_mnist_images(TRAIN_IMG_PATH, &data.images, &data.nImages);
    read_mnist_labels(TRAIN_LBL_PATH, &data.labels, &data.nImages);

    shuffle_data(data.images, data.labels, data.nImages);

    int train_size = (int)(data.nImages * TRAIN_SPLIT);
    int test_size = data.nImages - train_size;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        start = clock();
        for (int i = 0; i < train_size; i += MINI_BATCH_SIZE) {
            for(int j = 0; j < MINI_BATCH_SIZE; j++) {
                for (int k = 0; k < INPUT_SIZE; k++) {
                    img[j][k] = data.images[i * INPUT_SIZE + k] / 255.0f;
                }
            }
            for(int j = 0; j < MINI_BATCH_SIZE; j++) {
                labels[j] = data.labels[i + j];
            }
            train(&net, img, labels, learning_rate);
        }
        int correct = 0;
        for (int i = train_size; i < data.nImages; i++) {
            for (int k = 0; k < INPUT_SIZE; k++)
                img2[k] = data.images[i * INPUT_SIZE + k] / 255.0f;
            if (predict(&net, img2) == data.labels[i])
                correct++;
        }
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

        printf("Epoch %d, Accuracy: %.2f%%, Time: %.2f seconds\n", 
               epoch + 1, (float)correct / test_size * 100, cpu_time_used);
    }

    free(net.hidden.weights);
    free(net.hidden.biases);
    free(net.hidden.weight_momentum);
    free(net.hidden.bias_momentum);
    free(net.output.weights);
    free(net.output.biases);
    free(net.output.weight_momentum);
    free(net.output.bias_momentum);
    free(data.images);
    free(data.labels);

    return 0;
}