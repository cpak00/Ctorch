#include <cblas.h>
#include <stdio.h>

int main()
{
  float A[6] = {1.0, 2.0, 1.0, -3.0, 4.0, -1.0};
  float B[6] = {1.0, 2.0, 1.0, -3.0, 4.0, -1.0};
  float C[9] = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 2, 1., A, 2, B, 3, 2., C, 3);

  for(int i=0; i<9; i++)
    printf("%f ", C[i]);
  printf("\n");
}
