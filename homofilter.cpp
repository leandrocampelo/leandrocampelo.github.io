#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace cv;
using namespace std;

int dft_M, dft_N;

int gamaH_slider = 20;
int gamaH_slider_max = 100;

int gamaL_slider = 4;
int gamaL_slider_max = 100;

int d0_slider = 8;
int d0_slider_max = 100;

int c_slider = 5;
int c_slider_max = 100;

Mat filter, temp;

// faz a troca dos quadrantes
void deslocaDFT(Mat& image);

// aplica o filtro homomórfico
void homoFilter(int, void*);

int main(int argvc, char** argv){

    Mat image, imageExib, padded, complexImage;

    Mat_<float> zeros;
    Mat_<float> realInput;

    vector<Mat> planos;

    // caractere de parada
    char key;

    // carrega a imagem passada como parâmetro em grayscale
    image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

    // mostra a imagem na tela
    imshow("Imagem original em grayscale", image);

    // identifica os tamanhos ótimos para cálculo do FFT
    dft_M = getOptimalDFTSize(image.rows);
    dft_N = getOptimalDFTSize(image.cols);

    // realiza o padding da imagem
    copyMakeBorder(image, padded, 0,
                   dft_M - image.rows, 0,
                   dft_N - image.cols,
                   BORDER_CONSTANT, Scalar::all(0));

    // parte imaginária da matriz complexa (preenchida com zeros)
    zeros = Mat_<float>::zeros(padded.size());

    // prepara a matriz complexa para ser preenchida
    complexImage = Mat(padded.size(), CV_32FC2, Scalar(0));

    // a função de transferência (filtro frequencial) deve ter o
    // mesmo tamanho e tipo da matriz complexa
    filter = complexImage.clone();

    namedWindow("Imagem filtrada");

    // criando os sliders dos parâmetros do filtro
    createTrackbar("gamaH x 10", "Imagem filtrada", &gamaH_slider, gamaH_slider_max);

    createTrackbar("gamaL x 10", "Imagem filtrada", &gamaL_slider, gamaL_slider_max);

    createTrackbar("D0 x 10", "Imagem filtrada", &d0_slider, d0_slider_max);

    createTrackbar("C x 10", "Imagem filtrada", &c_slider, c_slider_max);

  while(1){

    // realiza o padding da imagem
    copyMakeBorder(image, padded, 0,
                   dft_M - image.rows, 0,
                   dft_N - image.cols,
                   BORDER_CONSTANT, Scalar::all(0));


    // limpa o array de matrizes que vão compor a imagem complexa
    planos.clear();

    // calcula o log
    cv::log(realInput,realInput);

    // cria a componente real
    realInput = Mat_<float>(padded);

    // insere as duas componentes no array de matrizes
    planos.push_back(realInput);
    planos.push_back(zeros);

    // combina o array de matrizes em uma única componente complexa
    merge(planos, complexImage);

    // calcula o dft
    dft(complexImage, complexImage);

    // realiza a troca de quadrantes
    deslocaDFT(complexImage);

    // aplica o filtro homomórfico
    homoFilter(gamaH_slider, 0);

    // faz a multiplicação por cada elemento das duas matrizes
    mulSpectrums(complexImage, filter, complexImage, 0);

    // troca novamente os quadrantes da matriz
    deslocaDFT(complexImage);

    // calcula a transformadainversa de Fourier
    idft(complexImage, complexImage);

    // limpa o array planos
    planos.clear();

    // separa a parte real e imaginária da matriz filtrada
    split (complexImage, planos);

    // normaliza a parte real para a exibição
    normalize(planos[0], planos[0], 0, 1, CV_MINMAX);

    // calcula a exponencial
    cv::exp(planos[0],planos[0]);

    // normaliza a parte real para a exibição
    normalize(planos[0], planos[0], 0, 1, CV_MINMAX);

    // mostra a imagem filtrada na tela
    imshow("Imagem filtrada", planos[0]);

    // condição para parar o algoritmo (pressionar a tecla esc)
    key = (char) waitKey(10);
    if( key == 27 ) break;
  }
    // grava a imagem
    imwrite("Imagem_em_grayscale.png", image);

    return 0;
}

// troca os quadrantes da imagem da DFT
void deslocaDFT(Mat& image){
    Mat tmp, A, B, C, D;

    // se a imagem tiver tamanho ímpar, recorta a região para
    // evitar cópias de tamanho desigual
    image = image(Rect(0, 0, image.cols & -2, image.rows & -2));
    int cx = image.cols/2;
    int cy = image.rows/2;

    // reorganiza os quadrantes da transformada
    // A B  ->   D C
    // C D       B A
    A = image(Rect(0, 0, cx, cy));
    B = image(Rect(cx, 0, cx, cy));
    C = image(Rect(0, cy, cx, cy));
    D = image(Rect(cx, cy, cx, cy));

    // A <-> D
    A.copyTo(tmp);  D.copyTo(A);  tmp.copyTo(D);

    // C <-> B
    C.copyTo(tmp);  B.copyTo(C);  tmp.copyTo(B);
}

void homoFilter(int, void*){
    float d2, gamaH, gamaL, d0, c;
    float M, N;

    M = dft_M;
    N = dft_N;

    // parâmetros do filtro homomórfico
    gamaH = (float)gamaH_slider/10.0;
    gamaL = (float)gamaL_slider/10.0;
    d0    = (float)d0_slider/10.0;
    c     = (float)c_slider/10.0;

    //cout << "gamaH=" << gamaH << " gamaL=" << gamaL << " d0=" << d0 << " c=" << c << endl;

    // cria uma matriz temporária para criar as componentes real
    // e imaginária do filtro ideal
    temp = Mat(dft_M, dft_N, CV_32F);

    for(int i=0; i<dft_M ;i++){
        for(int j=0; j<dft_N ;j++){
            d2 = ((float)i-M/2.0)*((float)i-M/2.0) + ((float)j-N/2.0)*((float)j-N/2.0);
            temp.at<float>(i,j) = (gamaH-gamaL)*(1.0-exp(-1.0*(float)c*(d2/(d0*d0))))+ gamaL;
        }
    }

    // cria a matriz com as componentes do filtro e junta
    // ambas em uma matriz multicanal complexa
    Mat comps[] = {temp, temp};
    merge(comps, 2, filter);

    imshow("H(u,v)", temp);
}
