#pragma once
#include "utils.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <math.h>
#include <filesystem>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define STREAM_NUM 4
#define BATCH_SIZE 16

// ********************** Connect Component Labeling in CUDA *****************************
//Reference: A Parallel Approach to Object Identification in Large-scale Images (ICESS2016)
__global__ void imageLabel(float *bin, float *label);
__device__ int findRoot(int row, int col, int imgWidth, float *label);
__global__ void labelMerge(float *label, int div);
__global__ void reLabel(float *label);
bool getHostBuffer(float *hostBuffer, vector<Mat> cv2_imgList, int w, int h);

int main()
{
	int width, height;
	cudaEvent_t start, stop;
	float  elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	vector<cv::Mat> cv2_imgList;

	cudaStream_t stream[STREAM_NUM];
	for (int i = 0; i < STREAM_NUM; i++)
	{
		cudaStreamCreate(&stream[i]);
	}

	bool load_batch = loadBatchedInput("input_images", BATCH_SIZE, cv2_imgList, width, height);
	assert(load_batch);

	float *inputs;
	float *d_inputs;
	float *d_label;
	float *h_label;
	cudaMallocHost((void**)&inputs, cv2_imgList.size()*width*height * sizeof(float));
	cudaMallocHost((void**)&h_label, cv2_imgList.size()*width*height * sizeof(float));
	cudaMalloc((void**)&d_inputs, cv2_imgList.size()*width*height * sizeof(float));
	cudaMalloc((void**)&d_label, cv2_imgList.size()*width*height * sizeof(float));
	
	bool get_host_buffer = getHostBuffer(inputs, cv2_imgList, width, height);

	dim3 dimBlock(width, 1, 1);
	dim3 dimGrid(BATCH_SIZE, height, 1);

	dim3 block(32, 32);
	dim3 grid(ceil((float)width / block.x), ceil((float)height / block.y));
	
	
	for (int i = 0; i < (int)(cv2_imgList.size() / BATCH_SIZE); i++)
	{
		cudaMemcpyAsync(d_inputs + i*BATCH_SIZE*width*height, inputs + i*BATCH_SIZE*width*height, BATCH_SIZE*width*height*sizeof(float), cudaMemcpyHostToDevice, stream[i%STREAM_NUM]);
		imageLabel << <dimGrid, dimBlock, 0, stream[i%STREAM_NUM] >> > (d_inputs + i * BATCH_SIZE*width*height, d_label + i * BATCH_SIZE*width*height);
		if (cudaSuccess != cudaGetLastError()) printf("Error at imageLabel !\n");

		////////////// label merge & relabel /////////////
		int div = 2;
		int nIter = static_cast<int>(log2(width));
		for (int v = 0; v < nIter; v++) {
			dim3 dimBlock2(height / div, 1, 1);
			labelMerge << <dimGrid, dimBlock2, 0, stream[i%STREAM_NUM] >> > (d_label + i*BATCH_SIZE*width*height, div);
			if (cudaSuccess != cudaGetLastError()) printf("Error at labelMerge !\n");
			div = div * 2;
		}
		reLabel <<<dimGrid, dimBlock, 0, stream[i%STREAM_NUM]>>> (d_label + i*BATCH_SIZE*width*height);
		if (cudaSuccess != cudaGetLastError()) printf("Error at reLabel !\n");
		cudaMemcpyAsync(h_label + i*BATCH_SIZE*width*height, d_label + i*BATCH_SIZE*width*height, BATCH_SIZE*width*height*sizeof(float), cudaMemcpyDeviceToHost, stream[i%STREAM_NUM]);
	}
	cudaDeviceSynchronize();

	bool label_out = saveLabels_array(h_label, width, height, "input_images", BATCH_SIZE);

	//for (int i = 0; i < width*height; i++)
	//{
	//	cout << (int)inputs[i] << " ";
	//	cout << (int)h_label[i] << " ";
	//	if (i > 0 && i % width == 0) cout << endl;
	//}

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;

}


bool getHostBuffer(float *hostBuffer, vector<Mat> cv2_imgList, int w, int h)
{
	for (int i = 0; i < cv2_imgList.size(); i++)
	{
		for (int j = 0; j < w*h; j++) {
			float* temp = (float*)cv2_imgList[i].data;
			hostBuffer[i*w*h + j] = temp[j];
		}
		//for (int y = 0; y < h; y++)
		//{
		//	for (int x = 0; x < w; x++)
		//	{
		//		hostBuffer[i*w*h + (y*w+x)] = cv2_imgList[i].at<float>(Point(x, y));
		//	} 
		//}
	}
	return true;
}

// ********************** Connect Component Labeling in CUDA *****************************
//Reference: A Parallel Approach to Object Identification in Large-scale Images (ICESS2016)
__global__ void imageLabel(float *bin, float *label) {
	int colIdx = threadIdx.x, rowIdx = blockIdx.y, imgIdx = blockIdx.x;
	int NCOL = blockDim.x, NROW = gridDim.y;
	int thread_1D_pos = blockIdx.x * NROW * NCOL + blockIdx.y * NCOL + threadIdx.x;

	label[thread_1D_pos] = thread_1D_pos;
	if ((int)bin[thread_1D_pos] < 1) label[thread_1D_pos] = -1.0;

	for (int i = NROW - 1; i > 0; i--) {
		int upperRow = imgIdx * NCOL * NROW + i * NCOL + colIdx;
		int underRow = imgIdx * NCOL * NROW + (i + 1) * NCOL + colIdx;
		if (bin[upperRow] > 0.0 && bin[underRow] > 0.0) {
			label[upperRow] = label[underRow];
		}
	}
}

__device__ int findRoot(int row, int col, int imgWidth, float *label) {
	if (label[imgWidth*row + col] < 0) {
		return -1.0;
	}

	int rootLabel = imgWidth * row + col;

	while (label[rootLabel] != rootLabel) {
		rootLabel = label[rootLabel];
	}

	return rootLabel;
}

__global__ void labelMerge(float *label, int div) {
	int NCOL = blockDim.x, NROW = gridDim.y;
	int thread_1D_pos = blockIdx.x * NROW * NCOL + blockIdx.y * NCOL + threadIdx.x;

	int imgWidth = NCOL * div;
	int nBoundary = imgWidth / div;

	int col = (thread_1D_pos % nBoundary) * div + div / 2 - 1;
	int row = static_cast<int>(thread_1D_pos / nBoundary);

	if (label[imgWidth*row + col] > 0 && label[imgWidth*row + col + 1] > 0)
	{
		int rootL = findRoot(row, col, imgWidth, label);
		int rootR = findRoot(row, col + 1, imgWidth, label);
		label[min(rootL, rootR)] = label[max(rootL, rootR)];
	}
}

__global__ void reLabel(float *label) {
	int NCOL = blockDim.x, NROW = gridDim.y;
	int thread_1D_pos = blockIdx.x * NROW * NCOL + blockIdx.y * NCOL + threadIdx.x;
	int row = static_cast<int>(thread_1D_pos / NCOL);
	int col = thread_1D_pos % NCOL;

	label[row*NCOL + col] = max(float(findRoot(row, col, NCOL, label)%NCOL*NROW), -1.0f);
	//label[row*NCOL + col] = (float)findRoot(row, col, NCOL, label);
	__syncthreads();
}