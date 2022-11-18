#include <iostream>
#include <fstream>
#include <omp.h>
#include <cstring>
#include <string>
#include "BMPFileRW.h"

using namespace std;

#pragma warning(disable : 4996)
typedef double(*TestFunctTemp1)(RGBTRIPLE**& rgb_in, RGBTRIPLE**& rgb_out, int& height, int& width, int& ksize);

void InsertSort(int* arr, int i, int length, int half) {
    int temp = 0;
    int j = 0;

    for (int f = half + i; f < length; f = f + half)
    {
        j = f;
        while (j > i && arr[j - half] > arr[j])
        {
            temp = arr[j];
            arr[j] = arr[j - half];
            arr[j - half] = temp;
            j = j - half;
        }
    }
}

double shellSort(int*& array, int& N) {
    double time_start = omp_get_wtime();

    int i, j, step;
    int tmp;
    for (step = N / 2; step > 0; step /= 2)
        for (i = 0; i < step; i++)
        {
            InsertSort(array, i, N, step);
        }
    double time_stop = omp_get_wtime();
    return time_stop - time_start;
}
void quickSort(int* array, int N) {
    long i = 0, j = N;
    int temp, p;

    p = array[N >> 1];

    do {
        while (array[i] < p) i++;
        while (array[j] > p) j--;

        if (i <= j) {
            temp = array[i]; array[i] = array[j]; array[j] = temp;
            i++; j--;
        }
    } while (i <= j);

    if (j > 0) quickSort(array, j);
    if (N > i) quickSort(array + i, N - i);
}

void fillMEDMAS(int* MEDMAS_R, int* MEDMAS_G, int* MEDMAS_B, RGBTRIPLE** rgb_in, int height, int width, int y, int x, int RH, int RW) {
    int masind = 0;

    for (int dy = -RH; dy <= RH; dy++) {
        int ky = y + dy;

        if (ky < 0)
            ky = 0;
        if (ky > height - 1)
            ky = height - 1;

        for (int dx = -RW; dx <= RW; dx++) {
            int kx = x + dx;

            if (kx < 0)
                kx = 0;
            if (kx > width - 1)
                kx = width - 1;

            MEDMAS_R[masind] = rgb_in[ky][kx].rgbtRed;
            MEDMAS_G[masind] = rgb_in[ky][kx].rgbtGreen;
            MEDMAS_B[masind] = rgb_in[ky][kx].rgbtBlue;
            masind++;
        }
    }
}

double medianShellFilter(RGBTRIPLE** rgb_in, RGBTRIPLE** rgb_out, int height, int width, int ksize) {
    double time_start = omp_get_wtime();

    int size = ksize * ksize;
    int RH = ksize / 2, RW = ksize / 2;

    int* MEDMAS_R = new int[size];
    int* MEDMAS_G = new int[size];
    int* MEDMAS_B = new int[size];

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            fillMEDMAS(MEDMAS_R, MEDMAS_G, MEDMAS_B, rgb_in, height, width, y, x, RH, RW);
            shellSort(MEDMAS_R, size);
            rgb_out[y][x].rgbtRed = MEDMAS_R[size / 2];
            shellSort(MEDMAS_G, size);
            rgb_out[y][x].rgbtGreen = MEDMAS_G[size / 2];
            shellSort(MEDMAS_B, size);
            rgb_out[y][x].rgbtBlue = MEDMAS_B[size / 2];

        }
    }
    delete[] MEDMAS_R;
    delete[] MEDMAS_G;
    delete[] MEDMAS_B;

    double time_stop = omp_get_wtime();
    return time_stop - time_start;
}

double medianShellFilterParallelFor(RGBTRIPLE** rgb_in, RGBTRIPLE** rgb_out, int height, int width, int ksize) {
    double time_start = omp_get_wtime();

    int size = ksize * ksize;
    int RH = ksize / 2, RW = ksize / 2;

#pragma omp parallel for
        for (int y = 0; y < height; ++y)
        {
            int* MEDMAS_R = new int[size];
            int* MEDMAS_G = new int[size];
            int* MEDMAS_B = new int[size];

            for (int x = 0; x < width; ++x)
            {
                fillMEDMAS(MEDMAS_R, MEDMAS_G, MEDMAS_B, rgb_in, height, width, y, x, RH, RW);
                shellSort(MEDMAS_R, size);
                rgb_out[y][x].rgbtRed = MEDMAS_R[size / 2];
                shellSort(MEDMAS_G, size);
                rgb_out[y][x].rgbtGreen = MEDMAS_G[size / 2];
                shellSort(MEDMAS_B, size);
                rgb_out[y][x].rgbtBlue = MEDMAS_B[size / 2];

            }

            delete[] MEDMAS_R;
            delete[] MEDMAS_G;
            delete[] MEDMAS_B;
        }

    double time_stop = omp_get_wtime();
    return time_stop - time_start;
}

double medianShellFilterSections(RGBTRIPLE** rgb_in, RGBTRIPLE** rgb_out, int height, int width, int ksize) {
    double time_start = omp_get_wtime();

    int size = ksize * ksize;
    int RH = ksize / 2, RW = ksize / 2;

    int p;
#pragma omp parallel
    {
        p = omp_get_num_threads();
    }

    int iteration1 = height / p;
    int iteration2 = height * 2 / p;
    int iteration3 = height * 3 / p;

#pragma omp parallel sections
    {
#pragma omp section
        {
            int* MEDMAS_R = new int[size];
            int* MEDMAS_G = new int[size];
            int* MEDMAS_B = new int[size];

            for (int y = 0; y < iteration1; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    fillMEDMAS(MEDMAS_R, MEDMAS_G, MEDMAS_B, rgb_in, height, width, y, x, RH, RW);
                    shellSort(MEDMAS_R, size);
                    rgb_out[y][x].rgbtRed = MEDMAS_R[size / 2];
                    shellSort(MEDMAS_G, size);
                    rgb_out[y][x].rgbtGreen = MEDMAS_G[size / 2];
                    shellSort(MEDMAS_B, size);
                    rgb_out[y][x].rgbtBlue = MEDMAS_B[size / 2];

                }
            }

            delete[] MEDMAS_R;
            delete[] MEDMAS_G;
            delete[] MEDMAS_B;
        }
#pragma omp section
        {
            if (p > 1)
            {
                int* MEDMAS_R = new int[size];
                int* MEDMAS_G = new int[size];
                int* MEDMAS_B = new int[size];

                for (int y = iteration1; y < iteration2; ++y)
                {
                    for (int x = 0; x < width; ++x)
                    {
                        fillMEDMAS(MEDMAS_R, MEDMAS_G, MEDMAS_B, rgb_in, height, width, y, x, RH, RW);
                        shellSort(MEDMAS_R, size);
                        rgb_out[y][x].rgbtRed = MEDMAS_R[size / 2];
                        shellSort(MEDMAS_G, size);
                        rgb_out[y][x].rgbtGreen = MEDMAS_G[size / 2];
                        shellSort(MEDMAS_B, size);
                        rgb_out[y][x].rgbtBlue = MEDMAS_B[size / 2];

                    }
                }

                delete[] MEDMAS_R;
                delete[] MEDMAS_G;
                delete[] MEDMAS_B;
            }
        }
#pragma omp section
        {
            if (p > 2)
            {
                int* MEDMAS_R = new int[size];
                int* MEDMAS_G = new int[size];
                int* MEDMAS_B = new int[size];

                for (int y = iteration2; y < iteration3; ++y)
                {
                    for (int x = 0; x < width; ++x)
                    {
                        fillMEDMAS(MEDMAS_R, MEDMAS_G, MEDMAS_B, rgb_in, height, width, y, x, RH, RW);
                        shellSort(MEDMAS_R, size);
                        rgb_out[y][x].rgbtRed = MEDMAS_R[size / 2];
                        shellSort(MEDMAS_G, size);
                        rgb_out[y][x].rgbtGreen = MEDMAS_G[size / 2];
                        shellSort(MEDMAS_B, size);
                        rgb_out[y][x].rgbtBlue = MEDMAS_B[size / 2];

                    }
                }

                delete[] MEDMAS_R;
                delete[] MEDMAS_G;
                delete[] MEDMAS_B;
            }
        }
#pragma omp section
        {
            if (p > 3)
            {
                int* MEDMAS_R = new int[size];
                int* MEDMAS_G = new int[size];
                int* MEDMAS_B = new int[size];

                for (int y = iteration3; y < height; ++y)
                {
                    for (int x = 0; x < width; ++x)
                    {
                        fillMEDMAS(MEDMAS_R, MEDMAS_G, MEDMAS_B, rgb_in, height, width, y, x, RH, RW);
                        shellSort(MEDMAS_R, size);
                        rgb_out[y][x].rgbtRed = MEDMAS_R[size / 2];
                        shellSort(MEDMAS_G, size);
                        rgb_out[y][x].rgbtGreen = MEDMAS_G[size / 2];
                        shellSort(MEDMAS_B, size);
                        rgb_out[y][x].rgbtBlue = MEDMAS_B[size / 2];

                    }
                }

                delete[] MEDMAS_R;
                delete[] MEDMAS_G;
                delete[] MEDMAS_B;
            }
        }
    }

    double time_stop = omp_get_wtime();
    return time_stop - time_start;
}

double medianQuickFilter(RGBTRIPLE** rgb_in, RGBTRIPLE** rgb_out, int height, int width, int ksize) {
    double time_start = omp_get_wtime();

    int size = ksize * ksize;
    int RH = ksize / 2, RW = ksize / 2;

    int* MEDMAS_R = new int[size];
    int* MEDMAS_G = new int[size];
    int* MEDMAS_B = new int[size];

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            fillMEDMAS(MEDMAS_R, MEDMAS_G, MEDMAS_B, rgb_in, height, width, y, x, RH, RW);
            quickSort(MEDMAS_R, size);
            rgb_out[y][x].rgbtRed = MEDMAS_R[size / 2];
            quickSort(MEDMAS_G, size);
            rgb_out[y][x].rgbtGreen = MEDMAS_G[size / 2];
            quickSort(MEDMAS_B, size);
            rgb_out[y][x].rgbtBlue = MEDMAS_B[size / 2];

        }
    }

    delete[] MEDMAS_R;
    delete[] MEDMAS_G;
    delete[] MEDMAS_B;

    double time_stop = omp_get_wtime();
    return time_stop - time_start;
}

double medianQuickFilterParallelFor(RGBTRIPLE** rgb_in, RGBTRIPLE** rgb_out, int height, int width, int ksize) {
    double time_start = omp_get_wtime();

    int size = ksize * ksize;
    int RH = ksize / 2, RW = ksize / 2;

#pragma omp parallel for
    for (int y = 0; y < height; ++y)
    {
        int* MEDMAS_R = new int[size];
        int* MEDMAS_G = new int[size];
        int* MEDMAS_B = new int[size];

        for (int x = 0; x < width; ++x)
        {
            fillMEDMAS(MEDMAS_R, MEDMAS_G, MEDMAS_B, rgb_in, height, width, y, x, RH, RW);
            quickSort(MEDMAS_R, size);
            rgb_out[y][x].rgbtRed = MEDMAS_R[size / 2];
            quickSort(MEDMAS_G, size);
            rgb_out[y][x].rgbtGreen = MEDMAS_G[size / 2];
            quickSort(MEDMAS_B, size);
            rgb_out[y][x].rgbtBlue = MEDMAS_B[size / 2];

        }

        delete[] MEDMAS_R;
        delete[] MEDMAS_G;
        delete[] MEDMAS_B;
    }

    double time_stop = omp_get_wtime();
    return time_stop - time_start;
}

double medianQuickFilterSections(RGBTRIPLE** rgb_in, RGBTRIPLE** rgb_out, int height, int width, int ksize) {
    double time_start = omp_get_wtime();

    int size = ksize * ksize;
    int RH = ksize / 2, RW = ksize / 2;

    int p;
#pragma omp parallel
    {
        p = omp_get_num_threads();
    }

    int iteration1 = height / p;
    int iteration2 = height * 2 / p;
    int iteration3 = height * 3 / p;

#pragma omp parallel sections 
    {
#pragma omp section
        {
            int* MEDMAS_R = new int[size];
            int* MEDMAS_G = new int[size];
            int* MEDMAS_B = new int[size];

            for (int y = 0; y < iteration1; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    fillMEDMAS(MEDMAS_R, MEDMAS_G, MEDMAS_B, rgb_in, height, width, y, x, RH, RW);
                    quickSort(MEDMAS_R, size);
                    rgb_out[y][x].rgbtRed = MEDMAS_R[size / 2];
                    quickSort(MEDMAS_G, size);
                    rgb_out[y][x].rgbtGreen = MEDMAS_G[size / 2];
                    quickSort(MEDMAS_B, size);
                    rgb_out[y][x].rgbtBlue = MEDMAS_B[size / 2];

                }
            }

            delete[] MEDMAS_R;
            delete[] MEDMAS_G;
            delete[] MEDMAS_B;
        }
#pragma omp section
        {
            if (p > 1)
            {
                int* MEDMAS_R = new int[size];
                int* MEDMAS_G = new int[size];
                int* MEDMAS_B = new int[size];

                for (int y = iteration1; y < iteration2; ++y)
                {
                    for (int x = 0; x < width; ++x)
                    {
                        fillMEDMAS(MEDMAS_R, MEDMAS_G, MEDMAS_B, rgb_in, height, width, y, x, RH, RW);
                        quickSort(MEDMAS_R, size);
                        rgb_out[y][x].rgbtRed = MEDMAS_R[size / 2];
                        quickSort(MEDMAS_G, size);
                        rgb_out[y][x].rgbtGreen = MEDMAS_G[size / 2];
                        quickSort(MEDMAS_B, size);
                        rgb_out[y][x].rgbtBlue = MEDMAS_B[size / 2];

                    }
                }

                delete[] MEDMAS_R;
                delete[] MEDMAS_G;
                delete[] MEDMAS_B;
            }
        }
#pragma omp section
        {
            if (p > 2)
            {
                int* MEDMAS_R = new int[size];
                int* MEDMAS_G = new int[size];
                int* MEDMAS_B = new int[size];

                for (int y = iteration2; y < iteration3; ++y)
                {
                    for (int x = 0; x < width; ++x)
                    {
                        fillMEDMAS(MEDMAS_R, MEDMAS_G, MEDMAS_B, rgb_in, height, width, y, x, RH, RW);
                        quickSort(MEDMAS_R, size);
                        rgb_out[y][x].rgbtRed = MEDMAS_R[size / 2];
                        quickSort(MEDMAS_G, size);
                        rgb_out[y][x].rgbtGreen = MEDMAS_G[size / 2];
                        quickSort(MEDMAS_B, size);
                        rgb_out[y][x].rgbtBlue = MEDMAS_B[size / 2];

                    }
                }

                delete[] MEDMAS_R;
                delete[] MEDMAS_G;
                delete[] MEDMAS_B;
            }
        }
#pragma omp section
        {
            if (p > 3)
            {
                int* MEDMAS_R = new int[size];
                int* MEDMAS_G = new int[size];
                int* MEDMAS_B = new int[size];

                for (int y = iteration3; y < height; ++y)
                {
                    for (int x = 0; x < width; ++x)
                    {
                        fillMEDMAS(MEDMAS_R, MEDMAS_G, MEDMAS_B, rgb_in, height, width, y, x, RH, RW);
                        quickSort(MEDMAS_R, size);
                        rgb_out[y][x].rgbtRed = MEDMAS_R[size / 2];
                        quickSort(MEDMAS_G, size);
                        rgb_out[y][x].rgbtGreen = MEDMAS_G[size / 2];
                        quickSort(MEDMAS_B, size);
                        rgb_out[y][x].rgbtBlue = MEDMAS_B[size / 2];

                    }
                }

                delete[] MEDMAS_R;
                delete[] MEDMAS_G;
                delete[] MEDMAS_B;
            }
        }
    }

    double time_stop = omp_get_wtime();
    return time_stop - time_start;
}

double testMedianShellFilter(RGBTRIPLE**& rgb_in, RGBTRIPLE**& rgb_out, int& height, int& width, int& ksize) {
    return medianShellFilter(rgb_in, rgb_out, height, width, ksize);
}
double testMedianQuickFilter(RGBTRIPLE**& rgb_in, RGBTRIPLE**& rgb_out, int& height, int& width, int& ksize) {
    return medianQuickFilter(rgb_in, rgb_out, height, width, ksize);
}
double testMedianShellFilterSections(RGBTRIPLE**& rgb_in, RGBTRIPLE**& rgb_out, int& height, int& width, int& ksize) {
    return medianShellFilterSections(rgb_in, rgb_out, height, width, ksize);
}
double testMedianQuickFilterSections(RGBTRIPLE**& rgb_in, RGBTRIPLE**& rgb_out, int& height, int& width, int& ksize) {
    return medianQuickFilterSections(rgb_in, rgb_out, height, width, ksize);
}
double testMedianShellFilterParallelFor(RGBTRIPLE**& rgb_in, RGBTRIPLE**& rgb_out, int& height, int& width, int& ksize) {
    return medianShellFilterParallelFor(rgb_in, rgb_out, height, width, ksize);
}
double testMedianQuickFilterParallelFor(RGBTRIPLE**& rgb_in, RGBTRIPLE**& rgb_out, int& height, int& width, int& ksize) {
    return medianQuickFilterParallelFor(rgb_in, rgb_out, height, width, ksize);
}

char* inBMP(int i) {
    string str;
    str = "c:\\temp\\input_X.bmp";
    char* cstr;
    switch (i)
    {
    case 1:
        str[14] = '1';
        break;
    case 2:
        str[14] = '2';
        break;
    case 3:
        str[14] = '3';
        break;
    case 4:
        str[14] = '4';
        break;
    default:
        break;
    }

    cstr = new char[str.length() + 1];
    strcpy(cstr, str.c_str());
    return cstr;
}
char* outBMP(int i, int alg) {
    string str;
    str = "c:\\temp\\output_X_Y.bmp";
    char* cstr;
    switch (i)
    {
    case 1:
        str[15] = '1';
        break;
    case 2:
        str[15] = '2';
        break;
    case 3:
        str[15] = '3';
        break;
    case 4:
        str[15] = '4';
        break;
    default:
        break;
    }

    switch (alg)
    {
    case 0:
        str[17] = '1';
        break;
    case 1:
        str[17] = '2';
        break;
    case 2:
        str[17] = '3';
        break;
    case 3:
        str[17] = '4';
        break;
    case 4:
        str[17] = '5';
        break;
    case 5:
        str[17] = '6';
        break;
    default:
        break;
    }
    cstr = new char[str.length() + 1];
    strcpy(cstr, str.c_str());
    cout << cstr << endl;
    return cstr;
}

double AvgTrustedInterval(double& avg, double*& times, int& cnt)
{
    double sd = 0, newAVg = 0;
    int newCnt = 0;
    for (int i = 0; i < cnt; i++)
    {
        sd += (times[i] - avg) * (times[i] - avg);
    }
    sd /= (cnt - 1.0);
    sd = sqrt(sd);
    for (int i = 0; i < cnt; i++)
    {
        if (avg - sd <= times[i] && times[i] <= avg + sd)
        {
            newAVg += times[i];
            newCnt++;
        }
    }
    if (newCnt == 0) newCnt = 1;
    return newAVg / newCnt;
}

double TestIter(void* Funct, RGBTRIPLE** rgb_in, int Height, int Width, int ksize, int iterations, int i, int alg)
{
    double curtime = 0, avgTime = 0, avgTimeT = 0, correctAVG = 0;;
    double* Times = new double[iterations];
    cout << endl;

    RGBTRIPLE** rgb_out;
    rgb_out = new RGBTRIPLE * [Height];
    rgb_out[0] = new RGBTRIPLE[Width * Height];
    for (int j = 1; j < Height; j++)
    {
        rgb_out[j] = &rgb_out[0][Width * j];
    }

    for (int j = 0; j < iterations; j++)
    {
        curtime = ((*(TestFunctTemp1)Funct)(rgb_in, rgb_out, Height, Width, ksize)) * 1000;
        Times[j] = curtime;
        avgTime += curtime;
        cout << "+";
    }
    cout << endl;

    avgTime /= iterations;
    cout << "AvgTime:" << avgTime << endl;

    avgTimeT = AvgTrustedInterval(avgTime, Times, iterations);
    cout << "AvgTimeTrusted:" << avgTimeT << endl;

    char* cstr = outBMP(i, alg);

    BMPWrite(rgb_out, Width, Height, cstr);
    delete[] rgb_out[0];
    delete[] rgb_out;
    delete[] cstr;
    return avgTimeT;
}

void test_functions(void** Functions, string(&function_names)[6])
{
    RGBTRIPLE** rgb_in;
    BITMAPFILEHEADER header;
    BITMAPINFOHEADER bmiHeader;
    int imWidth = 0, imHeight = 0;

    int iters = 2;
    int nd = 0;
    double times[3][6][3][3];
    for (int i = 1; i < 4; i++)
    {
        char* cstr = inBMP(i);
        BMPRead(rgb_in, header, bmiHeader, cstr);
        imWidth = bmiHeader.biWidth;
        imHeight = bmiHeader.biHeight;

        for (int threads = 1; threads <= 4; threads++)
        {
            omp_set_num_threads(threads);
            //перебор алгоритмов по условиям
            for (int alg = 0; alg < 6; alg++)
            {
                int ksize = 7;
                for (int j = 0; j < 3; j++)
                {
                    if (threads == 1)
                    {
                        if (alg == 0 || alg == 3) {
                            times[nd][alg][j][0] = TestIter(Functions[alg], rgb_in, imHeight, imWidth, ksize, iters, i, alg);
                            // iters - кол-во запусков алгоритма
                            times[nd][alg][j][1] = times[nd][alg][j][0];
                            times[nd][alg][j][2] = times[nd][alg][j][0];
                        }
                    }
                    else
                    {
                        if (alg != 0 && alg != 3)
                        {
                            times[nd][alg][j][threads - 2] = TestIter(Functions[alg], rgb_in, imHeight, imWidth, ksize, iters, i, alg);
                        }
                    }
                    ksize += 4;
                }
            }
        }
        delete[] cstr;
        nd++;
    }
    ofstream fout("output.txt");
    fout.imbue(locale("Russian"));
    for (int ND = 0; ND < 3; ND++)
    {
        switch (ND)
        {
        case 0:
            cout << "\n----------1280*720----------" << endl;
            break;
        case 1:
            cout << "\n----------1920*1080----------" << endl;
            break;
        case 2:
            cout << "\n----------2580*1080----------" << endl;
            break;
        case 3:
            cout << "\n----------3840*2160----------" << endl;
            break;
        default:
            break;
        }
        for (int alg = 0; alg < 6; alg++)
        {
        for (int threads = 1; threads <= 4; threads++)
        {
            cout << "Поток " << threads << " --------------" << endl;
                for (int j = 0; j < 3; j++)
                {
                    cout << "Ksize = " << j << " --------------" << endl;
                    if (threads == 1)
                    {
                        if (alg == 0 || alg == 3) {
                            cout << function_names[alg] << "\t" << times[ND][alg][j][0] << " ms." << endl;
                            fout << times[ND][alg][j][0] << endl;
                        }
                    }
                    else
                    {
                        if (alg != 0 && alg != 3)
                        {
                            cout << function_names[alg] << "\t" << times[ND][alg][j][threads - 2] << " ms." << endl;
                            fout << times[ND][alg][j][threads - 2] << endl;
                        }
                    }
                }
            }
        }
    }
    fout.close();
}

int main()
{
    setlocale(LC_ALL, "RUS");

    void** FunctionsINT = new void* [6]{ testMedianShellFilter, testMedianShellFilterSections, testMedianShellFilterParallelFor,
        testMedianQuickFilter, testMedianQuickFilterSections, testMedianQuickFilterParallelFor };
    string function_names[6]{ "медианная фильтрация(shell sort)", "медианная фильтрация(shell sort sections)",
        "медианная фильтрация(shell sort parallel for)", "медианная фильтрация(quick sort)", "медианная фильтрация(quick sort sections)",
        "медианная фильтрация(quick sort parallel for)" };
    test_functions(FunctionsINT, function_names);



	/*RGBTRIPLE** rgb_in, ** rgb_out;
	BITMAPFILEHEADER header;
	BITMAPINFOHEADER bmiHeader;
	int imWidth = 0, imHeight = 0; 
	BMPRead(rgb_in, header, bmiHeader, "c:\\temp\\input_2.bmp");
	imWidth = bmiHeader.biWidth;
	imHeight = bmiHeader.biHeight;
	std::cout << "Image params:" << imWidth << "x" << imHeight << std::endl;
	rgb_out = new RGBTRIPLE * [imHeight];
	rgb_out[0] = new RGBTRIPLE[imWidth * imHeight];
	for (int i = 1; i < imHeight; i++)
	{
		rgb_out[i] = &rgb_out[0][imWidth * i];
	}

    for (int threads = 2; threads <= 4; threads++)
    {
        omp_set_num_threads(threads);
        cout << medianShellFilterParallelFor(rgb_in, rgb_out, imHeight, imWidth, 7) << endl;
        cout << threads << endl;
    }
	BMPWrite(rgb_out, imWidth, imHeight, "c:\\temp\\test_copy.bmp");
	std::cout << "Image saved\n";*/

	return 0;
}