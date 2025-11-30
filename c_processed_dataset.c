#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N_FEATURES 10       // <= DEBE COINCIDIR con n_components del PCA en Python
#define MAX_LINE_LEN 8192   // tamaño máximo de una línea de CSV
#define LEARNING_RATE 0.1
#define EPOCHS 500

// ------------------- Funciones auxiliares -------------------

double sigmoid(double z) {
    // Sigmoide para regresión logística
    if (z < -50.0) return 1e-22; // evitar underflow
    if (z >  50.0) return 1.0 - 1e-22; // evitar overflow
    return 1.0 / (1.0 + exp(-z));
}

// Cuenta cuántas líneas (muestras) hay en un CSV
int count_lines(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error abriendo archivo %s\n", filename);
        exit(EXIT_FAILURE);
    }
    int lines = 0;
    char buffer[MAX_LINE_LEN];
    while (fgets(buffer, MAX_LINE_LEN, f) != NULL) {
        // ignorar líneas vacías
        if (buffer[0] == '\n' || buffer[0] == '\r') continue;
        lines++;
    }
    fclose(f);
    return lines;
}

// Carga una matriz de tamaño [n_samples x N_FEATURES] desde CSV sin header
double **load_matrix(const char *filename, int *out_n_samples) {
    int n_samples = count_lines(filename);
    *out_n_samples = n_samples;

    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error abriendo archivo %s\n", filename);
        exit(EXIT_FAILURE);
    }

    double **X = (double **)malloc(n_samples * sizeof(double *));
    if (!X) {
        fprintf(stderr, "Error de memoria para X\n");
        exit(EXIT_FAILURE);
    }

    char buffer[MAX_LINE_LEN];
    int row = 0;

    while (fgets(buffer, MAX_LINE_LEN, f) != NULL && row < n_samples) {
        // ignorar líneas vacías
        if (buffer[0] == '\n' || buffer[0] == '\r') continue;

        X[row] = (double *)malloc(N_FEATURES * sizeof(double));
        if (!X[row]) {
            fprintf(stderr, "Error de memoria para fila %d\n", row);
            exit(EXIT_FAILURE);
        }

        int col = 0;
        char *token = strtok(buffer, ",");
        while (token != NULL && col < N_FEATURES) {
            X[row][col] = atof(token);
            col++;
            token = strtok(NULL, ",");
        }

        if (col != N_FEATURES) {
            fprintf(stderr,
                    "Error: en %s, fila %d, esperadas %d columnas, leídas %d\n",
                    filename, row + 1, N_FEATURES, col);
            exit(EXIT_FAILURE);
        }

        row++;
    }

    fclose(f);
    return X;
}

// Carga un vector y (0/1) desde CSV de una sola columna sin header
double *load_vector(const char *filename, int expected_n) {
    int n_samples = count_lines(filename);
    if (n_samples != expected_n) {
        fprintf(stderr,
                "Error: %s contiene %d filas, pero se esperaban %d\n",
                filename, n_samples, expected_n);
        exit(EXIT_FAILURE);
    }

    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error abriendo archivo %s\n", filename);
        exit(EXIT_FAILURE);
    }

    double *y = (double *)malloc(n_samples * sizeof(double));
    if (!y) {
        fprintf(stderr, "Error de memoria para y\n");
        exit(EXIT_FAILURE);
    }

    char buffer[MAX_LINE_LEN];
    int i = 0;
    while (fgets(buffer, MAX_LINE_LEN, f) != NULL && i < n_samples) {
        // ignorar líneas vacías
        if (buffer[0] == '\n' || buffer[0] == '\r') continue;
        y[i] = atof(buffer);  // debería ser 0 o 1
        i++;
    }

    fclose(f);
    return y;
}

// Predicción de probabilidad para un ejemplo x
double predict_proba(double *x, double *w) {
    double z = 0.0;
    for (int j = 0; j < N_FEATURES; j++) {
        z += x[j] * w[j];
    }
    return sigmoid(z);
}

// Predicción de clase (0/1) dado x y w
int predict_class(double *x, double *w) {
    double p = predict_proba(x, w);
    return (p >= 0.5) ? 1 : 0;
}

// Entrenamiento de regresión logística con gradiente descendente
void train_logistic(
    double **X, double *y,
    int n_samples,
    double *w,
    double lr,
    int epochs
) {
    // Inicializar pesos en 0
    for (int j = 0; j < N_FEATURES; j++) {
        w[j] = 0.0;
    }

    double *grad = (double *)malloc(N_FEATURES * sizeof(double));
    if (!grad) {
        fprintf(stderr, "Error de memoria para gradiente\n");
        exit(EXIT_FAILURE);
    }

    for (int e = 0; e < epochs; e++) {
        // Reiniciar gradiente
        for (int j = 0; j < N_FEATURES; j++) {
            grad[j] = 0.0;
        }

        // Acumular gradiente sobre todas las muestras
        for (int i = 0; i < n_samples; i++) {
            double y_pred = predict_proba(X[i], w);
            double error = y_pred - y[i];  // derivada de la log-loss

            for (int j = 0; j < N_FEATURES; j++) {
                grad[j] += error * X[i][j];
            }
        }

        // Actualizar pesos
        for (int j = 0; j < N_FEATURES; j++) {
            w[j] -= lr * grad[j] / (double)n_samples;
        }

        // Imprimir pérdida aproximada cada cierto número de épocas
        if ((e + 1) % 100 == 0) {
            double loss = 0.0;
            for (int i = 0; i < n_samples; i++) {
                double p = predict_proba(X[i], w);
                // log-loss binaria (aprox, evitando log(0))
                p = fmax(fmin(p, 1.0 - 1e-15), 1e-15);
                loss += -(y[i] * log(p) + (1.0 - y[i]) * log(1.0 - p));
            }
            loss /= (double)n_samples;
            printf("Epoca %d, loss ~ %.6f\n", e + 1, loss);
        }
    }

    free(grad);
}

// Calcula accuracy sobre dataset
double accuracy(double **X, double *y, int n_samples, double *w) {
    int correct = 0;
    for (int i = 0; i < n_samples; i++) {
        int pred = predict_class(X[i], w);
        if ((int)y[i] == pred) {
            correct++;
        }
    }
    return (double)correct / (double)n_samples;
}


int main(void) {
    // Archivos generados por las celdas de preprocesamiento en Python
    const char *X_train_file = "X_train_pca.csv";
    const char *X_test_file  = "X_test_pca.csv";
    const char *y_train_file = "y_train.csv";
    const char *y_test_file  = "y_test.csv";

    int n_train, n_test;

    // Cargar matrices de características PCA
    double **X_train = load_matrix(X_train_file, &n_train);
    double **X_test  = load_matrix(X_test_file, &n_test);

    // Cargar vectores de etiquetas
    double *y_train = load_vector(y_train_file, n_train);
    double *y_test  = load_vector(y_test_file, n_test);

    printf("Muestras de entrenamiento: %d, características (PCA): %d\n",
           n_train, N_FEATURES);
    printf("Muestras de prueba: %d, características (PCA): %d\n",
           n_test, N_FEATURES);

    // Entrenar modelo de regresión logística
    double *w = (double *)malloc(N_FEATURES * sizeof(double));
    if (!w) {
        fprintf(stderr, "Error de memoria para pesos\n");
        exit(EXIT_FAILURE);
    }

    printf("Entrenando modelo...\n");
    train_logistic(X_train, y_train, n_train, w, LEARNING_RATE, EPOCHS);

    // Evaluar en train y test
    double acc_train = accuracy(X_train, y_train, n_train, w);
    double acc_test  = accuracy(X_test, y_test, n_test, w);

    printf("Accuracy en entrenamiento: %.4f\n", acc_train);
    printf("Accuracy en prueba:       %.4f\n", acc_test);

    // Ejemplo de predicción para la primera muestra de test
    int pred0 = predict_class(X_test[0], w);
    double p0 = predict_proba(X_test[0], w);
    printf("Ejemplo: primera muestra de test -> y_real = %d, y_pred = %d, p_yes = %.4f\n",
           (int)y_test[0], pred0, p0);

    // Liberar memoria
    for (int i = 0; i < n_train; i++) free(X_train[i]);
    for (int i = 0; i < n_test; i++)  free(X_test[i]);
    free(X_train);
    free(X_test);
    free(y_train);
    free(y_test);
    free(w);

    return 0;
}


