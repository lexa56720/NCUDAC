
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cublas_v2.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cblas.h>
#include <omp.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
void printMatrixCuda(float* matrix, int rows, int columns)
{
	float* array = (float*)malloc(rows * columns * 4);
	cudaMemcpy(array, matrix, rows * columns * 4, cudaMemcpyDeviceToHost);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < columns; j++)
		{
			printf("%f ", array[j + i * columns]);
		}
		printf("\n");
	}
	printf("\n");
}
void GetThreadAndBlocks(int count, int* threads, int* blocks)
{
	*blocks = ceil((float)count / 1024);
	if (blocks == 0)
		*blocks = 1;
	*threads = count;
	if (*threads > 1024)
		*threads = 1024;
}


void MultiplyCuda(int rowsA, int columnsA, int columnsB, float* matrixA, float* matrixB, float* result, cublasOperation_t OpA, cublasOperation_t OpB, float scalar, cublasHandle_t handle)
{
	float alpha = scalar;
	float beta = 0.0f;
	// Запуск видеокарты с одним потоком для каждого элемента.
	cublasSgemm(handle, OpA, OpB,
		columnsB, rowsA, columnsA,
		&alpha,
		matrixB, columnsB,
		matrixA, columnsA,
		&beta,
		result, columnsB);

	// Ожидание конца вычислений
	cudaDeviceSynchronize();
}


extern "C"
{
	struct CudaData
	{
		cublasHandle_t handler;

		float speed;
		float momentum;


		float* deviceResult;
		float* deviceNeuronOutputs;
		float* deviceWeights;
		float* deviceIdeal;


		float* prevDeltas;
		float* deltas;

		float* deltasBuffer;
		float* weightBufferA;
		float* weightBufferB;
		float* layerBufferA;
		float* layerBufferB;
		float* layerBufferC;

		int maxVectorSizeTotal;
		int weightsSizeTotal;

		int inputSizeTotal;
		int resultSizeTotal;

		int* neuronsPerLayer;
		int layerCount;
		int neuronsSizeTotal;
	};


	void printMatrix(float* matrix, int rows, int columns)
	{
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < columns; j++)
			{
				printf("%f ", matrix[j + i * columns]);
			}
			printf("\n");
		}
		printf("\n");
	}


	int GetMax(int* array, int size)
	{
		int max = 0;
		for (int i = 0; i < size; i++)
			if (array[i] > max)
				max = array[i];
		return max;
	}

	int GetMaxWeightCount(int* neuronsPerLayer, int layerCount)
	{
		int max = 0;
		for (int i = 0; i < layerCount - 1; i++)
			if ((neuronsPerLayer[i] + 1) * neuronsPerLayer[i + 1] > max)
				max = (neuronsPerLayer[i] + 1) * neuronsPerLayer[i + 1];
		return max;
	}

	int TotalWeightCount(int* neurons, int* layerCount)
	{
		int count = 0;
		for (int i = 0; i < (*layerCount - 1); i++)
			count += (neurons[i] + 1) * neurons[i + 1];
		return count;
	}

	int ArraySum(int* array, int size)
	{
		int sum = 0;
		for (int i = 0; i < size; i++)
			sum += array[i];
		return sum;
	}

	__global__ void MultiplyScalarKernel(float* result, float* matrix, float scalar, int count)
	{
		int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadIndex < count)
		{
			result[threadIndex] = matrix[threadIndex] * scalar;
		}

	}
	void MultiplyScalarCuda(float* matrix, float scalar, int rows, int columns, float* result)
	{
		int count = rows * columns;
		int blocks, threads;
		GetThreadAndBlocks(count, &threads, &blocks);
		MultiplyScalarKernel << <blocks, threads >> > (result, matrix, scalar, count);

		// Ожидание конца вычислений
		cudaDeviceSynchronize();
	}

	__global__ void ActivateKernel(float* matrix, int count)
	{
		int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadIndex < count)
		{
			matrix[threadIndex] = 1.0 / (1.0 + expf(-matrix[threadIndex]));
		}
	}
	void MultiplyAndActivateCublas(int rows, int columns, float* matrix, float* vector, float* result, cublasHandle_t handler)
	{
		//Определение количества потоков и блоков
		int blocks, threads;
		GetThreadAndBlocks(rows, &threads, &blocks);
		float alpha = 1.0f;
		float beta = 0.0f;

		cublasSgemv(handler, CUBLAS_OP_T, columns, rows, &alpha, matrix, columns, vector, 1, &beta, result, 1);

		// Ожидание конца вычислений
		cudaDeviceSynchronize();
		// Запуск видеокарты с одним потоком для каждого элемента.
		ActivateKernel << <blocks, threads >> > (result, rows);

		// Ожидание конца вычислений
		cudaDeviceSynchronize();
	}

	__global__ void SubstractKernel(float* a, float* b, float* result, int count)
	{
		int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadIndex < count)
		{
			result[threadIndex] = a[threadIndex] - b[threadIndex];
		}
	}
	void SubstractCuda(float* a, float* b, float* result, int count)
	{
		int blocks, threads;
		GetThreadAndBlocks(count, &threads, &blocks);
		SubstractKernel << <blocks, threads >> > (a, b, result, count);

		// Ожидание конца вычислений
		cudaDeviceSynchronize();
	}

	__global__ void AddKernel(float* a, float* b, float* result, int count)
	{
		int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadIndex < count)
		{
			result[threadIndex] = a[threadIndex] + b[threadIndex];
		}
	}
	void AddCuda(float* a, float* b, float* result, int count)
	{
		int blocks, threads;
		GetThreadAndBlocks(count, &threads, &blocks);
		AddKernel << <blocks, threads >> > (a, b, result, count);

		// Ожидание конца вычислений
		cudaDeviceSynchronize();
	}

	__global__ void DeActivateKernel(float* neuronsOutput, float* result, int count)
	{
		int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadIndex < count)
		{
			result[threadIndex] = (1 - neuronsOutput[threadIndex]) * neuronsOutput[threadIndex];
		}
	}
	void DeActivateCuda(float* neuronsOutput, float* result, int count)
	{
		int blocks, threads;
		GetThreadAndBlocks(count, &threads, &blocks);
		DeActivateKernel << <blocks, threads >> > (neuronsOutput, result, count);

		// Ожидание конца вычислений
		cudaDeviceSynchronize();
	}

	__global__ void HadamardProductKernel(float* a, float* b, float* result, int count)
	{
		int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadIndex < count)
		{
			result[threadIndex] = a[threadIndex] * b[threadIndex];
		}
	}
	void HadamardProductCuda(float* a, float* b, float* result, int rows, int columns)
	{
		int count = rows * columns;
		int blocks, threads;
		GetThreadAndBlocks(count, &threads, &blocks);
		HadamardProductKernel << <blocks, threads >> > (a, b, result, count);

		// Ожидание конца вычислений
		cudaDeviceSynchronize();
	}

	__global__ void TransposeKernel(float* matrix, int rows, int columns, int count, float* result)
	{
		int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadIndex < count)
		{
			int row = (int)(threadIndex / columns);
			int column = threadIndex % columns;
			result[threadIndex] = matrix[column * rows + row];
		}
	}
	void TransposeCuda(float* matrix, int rows, int columns, float* result)
	{
		int count = rows * columns;
		int blocks, threads;
		GetThreadAndBlocks(count, &threads, &blocks);

		TransposeKernel << <blocks, threads >> > (matrix, rows, columns, count, result);

		// Ожидание конца вычислений
		cudaDeviceSynchronize();
	}

	__global__ void MemsetKernel(float* array, float value, int count)
	{
		int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadIndex < count)
		{
			array[threadIndex] = value;
		}
	}
	void MemsetCuda(float* array, float value, int count)
	{
		int blocks, threads;
		GetThreadAndBlocks(count, &threads, &blocks);
		MemsetKernel << <blocks, threads >> > (array, value, count);

		// Ожидание конца вычислений
		cudaDeviceSynchronize();
	}


	__global__ void SumToSingleColumnKernel(float* matrix, int columns, int rows, float* result)
	{
		int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

		if (threadIndex < rows)
		{
			result[threadIndex] = 0;
			for (int i = 0; i < columns; i++)
				result[threadIndex] += matrix[threadIndex * columns + i];
		}
	}
	void SumToSingleColumnCuda(float* matrix, int rows, int columns, float* result)
	{
		int blocks, threads;
		GetThreadAndBlocks(rows, &threads, &blocks);
		SumToSingleColumnKernel << <blocks, threads >> > (matrix, columns, rows, result);

		// Ожидание конца вычислений
		cudaDeviceSynchronize();
	}
	void SumToSingleColumn(float* matrix, int rows, int columns, float* result)
	{
		for (int i = 0; i < rows; i++)
			for (int j = 0; j < columns; j++)
			{
				result[i] += matrix[i * columns + j];
			}

		// Ожидание конца вычислений
		cudaDeviceSynchronize();
	}

	__declspec(dllexport) void GetStructSize(int* size)
	{
		*size = sizeof(struct CudaData);
	}

	__declspec(dllexport) void CudaMalloc(float* weights,
		int* neurons, int* layerCount, float* speed, float* momentum, char* result)
	{
		struct CudaData* data = (CudaData*)malloc(sizeof(CudaData));;

		cublasCreate(&data->handler);
		int* neuronsPerLayer = (int*)malloc(*layerCount * sizeof(int));
		memcpy(neuronsPerLayer, neurons, *layerCount * sizeof(int));

		data->inputSizeTotal = neurons[0] * sizeof(float);
		data->weightsSizeTotal = TotalWeightCount(neurons, layerCount) * sizeof(float);
		data->resultSizeTotal = neurons[*layerCount - 1] * sizeof(float);
		data->neuronsSizeTotal = (ArraySum(neurons, *layerCount) + (*layerCount - 1)) * sizeof(float);
		data->neuronsPerLayer = neuronsPerLayer;
		data->layerCount = *layerCount;
		data->maxVectorSizeTotal = (GetMax(neurons, *layerCount) + 1) * sizeof(float);

		data->speed = *speed;
		data->momentum = *momentum;

		int maxLayerWeight = GetMaxWeightCount(neurons, *layerCount) * sizeof(float);

		// Выделение памяти и инициализация данных
		cudaMalloc((void**)&data->deviceIdeal, data->inputSizeTotal); //Массив идеальных значений
		cudaMalloc((void**)&data->deviceResult, data->resultSizeTotal); //Массив входных данных
		cudaMalloc((void**)&data->deviceWeights, data->weightsSizeTotal); //Массив всех весов
		cudaMalloc((void**)&data->deltas, data->neuronsSizeTotal); //Массив всех весов
		cudaMalloc((void**)&data->deltasBuffer, data->neuronsSizeTotal);
		cudaMalloc((void**)&data->prevDeltas, data->weightsSizeTotal); //Массив всех весов
		cudaMalloc((void**)&data->deviceNeuronOutputs, data->neuronsSizeTotal); //Входные данные следующего слоя
		cudaMalloc((void**)&data->layerBufferA, data->maxVectorSizeTotal); //Входные данные следующего слоя
		cudaMalloc((void**)&data->layerBufferB, data->maxVectorSizeTotal); //Входные данные следующего слоя
		cudaMalloc((void**)&data->layerBufferC, data->maxVectorSizeTotal); //Входные данные следующего слоя

		cudaMalloc((void**)&data->weightBufferA, maxLayerWeight);
		cudaMalloc((void**)&data->weightBufferB, maxLayerWeight);


		MemsetCuda(data->deltas, 0, data->neuronsSizeTotal / sizeof(float));
		MemsetCuda(data->prevDeltas, 0, data->neuronsSizeTotal / sizeof(float));
		MemsetCuda(data->deviceNeuronOutputs, 1.0, data->neuronsSizeTotal / sizeof(float));

		cudaMemcpy(data->deviceWeights, weights, data->weightsSizeTotal, cudaMemcpyHostToDevice);

		memcpy(result, data, sizeof(CudaData));
		free(data);
	}


	void UpdateWeights(struct CudaData* data)
	{
		int weightPosition = 0;
		int neuronsPosition = data->neuronsPerLayer[0] + 1;
		int prevNeronsPosition = 0;
		for (int layer = 1; layer < data->layerCount; layer++)
		{

			MultiplyCuda(
				data->neuronsPerLayer[layer],
				1,
				data->neuronsPerLayer[layer - 1] + 1,
				&data->deltas[neuronsPosition],
				&data->deviceNeuronOutputs[prevNeronsPosition],
				data->weightBufferA,
				CUBLAS_OP_N,
				CUBLAS_OP_N,
				data->speed,
				data->handler
			);


			MultiplyScalarCuda(
				&data->prevDeltas[weightPosition],
				data->momentum,
				data->neuronsPerLayer[layer],
				data->neuronsPerLayer[layer - 1] + 1,
				data->weightBufferB
			);

			AddCuda
			(
				data->weightBufferA,
				data->weightBufferB,
				&data->prevDeltas[weightPosition],
				data->neuronsPerLayer[layer] * (data->neuronsPerLayer[layer - 1] + 1)
			);

			AddCuda(
				&data->deviceWeights[weightPosition],
				&data->prevDeltas[weightPosition],
				&data->deviceWeights[weightPosition],
				(data->neuronsPerLayer[layer]) * (data->neuronsPerLayer[layer - 1] + 1)
			);

			prevNeronsPosition = neuronsPosition;
			weightPosition += (data->neuronsPerLayer[layer - 1] + 1) * data->neuronsPerLayer[layer];
			neuronsPosition += data->neuronsPerLayer[layer] + 1;
		}
	}

	void ComputeDeltas(struct CudaData* data, int weightPosition, int neuronPosition)
	{
		int neuronsOnLastLayer = data->neuronsPerLayer[data->layerCount - 1];
		int neuronsOnPreLastLayer = data->neuronsPerLayer[data->layerCount - 2] + 1;


		weightPosition -= neuronsOnPreLastLayer * neuronsOnLastLayer;
		neuronPosition -= neuronsOnLastLayer;

		SubstractCuda(
			data->deviceIdeal,
			&data->deviceNeuronOutputs[neuronPosition],
			data->layerBufferB,
			neuronsOnLastLayer);


		DeActivateCuda(
			&data->deviceNeuronOutputs[neuronPosition],
			data->layerBufferA,
			neuronsOnLastLayer);


		HadamardProductCuda(
			data->layerBufferB,
			data->layerBufferA,
			&data->deltas[neuronPosition],
			neuronsOnLastLayer,
			1
		);

		neuronPosition -= neuronsOnPreLastLayer;

		for (int layer = (data->layerCount - 2); layer > 0; layer--)
		{
			DeActivateCuda(
				&data->deviceNeuronOutputs[neuronPosition],
				data->layerBufferA,
				data->neuronsPerLayer[layer]);

			MultiplyCuda(
				data->neuronsPerLayer[layer] + 1,
				data->neuronsPerLayer[layer + 1],
				1,
				&data->deviceWeights[weightPosition],
				&data->deltas[neuronPosition + data->neuronsPerLayer[layer] + 1],
				data->layerBufferB,
				CUBLAS_OP_N,
				CUBLAS_OP_N,
				1,
				data->handler);

			HadamardProductCuda(
				data->layerBufferB,
				data->layerBufferA,
				&data->deltas[neuronPosition],
				data->neuronsPerLayer[layer] + 1,
				1);

			neuronPosition -= (data->neuronsPerLayer[layer] + 1);
			weightPosition -= data->neuronsPerLayer[layer + 1] * data->neuronsPerLayer[layer] + 1;
		}
	}

	void LearnCuda(struct CudaData* data, int weightPosition, int neuronPosition)
	{
		ComputeDeltas(data, weightPosition, neuronPosition);
		//printMatrixCuda(data->deltas, 1, data->neuronsSizeTotal / 4);
		UpdateWeights(data);
	}


	__declspec(dllexport) void CudaExecute(float* input, float* result, struct CudaData data)
	{
		cudaMemcpy(data.deviceNeuronOutputs, input, data.inputSizeTotal, cudaMemcpyHostToDevice);

		//Устанавливаем позицию в массиве весов
		int weightPosition = 0;
		int neuronsPosition = data.neuronsPerLayer[0] + 1;

		float* inputPtr = &data.deviceNeuronOutputs[0];
		for (int layer = 0; layer < (data.layerCount - 1); layer++)
		{
			//Умножаем матрицу весов на вектор входных значений слоя
			MultiplyAndActivateCublas(
				data.neuronsPerLayer[layer + 1],
				data.neuronsPerLayer[layer] + 1,
				&data.deviceWeights[weightPosition],
				inputPtr,
				&data.deviceNeuronOutputs[neuronsPosition],
				data.handler);

			//Свапаем указатели
			inputPtr = &data.deviceNeuronOutputs[neuronsPosition];

			//Обновляем положение в массиве весов
			weightPosition += (data.neuronsPerLayer[layer] + 1) * data.neuronsPerLayer[layer + 1];
			neuronsPosition += data.neuronsPerLayer[layer + 1] + 1;
		}
		neuronsPosition--;

		cudaMemcpy(
			result,
			&data.deviceNeuronOutputs[neuronsPosition - data.neuronsPerLayer[data.layerCount - 1]],
			data.resultSizeTotal,
			cudaMemcpyDeviceToHost);
	}


	__declspec(dllexport) void CudaExecuteAndLearnBatch(float* inputs, float* ideals, int* batchSize, float* result, struct CudaData data)
	{

		int neuronsOnLastLayer = data.neuronsPerLayer[data.layerCount - 1];

		MemsetCuda(data.deltasBuffer, 0.0f, data.neuronsSizeTotal / sizeof(float));
		for (int i = 0; i < *batchSize; i++)
		{
			cudaMemcpy(data.deviceIdeal, &ideals[i * neuronsOnLastLayer], data.resultSizeTotal, cudaMemcpyHostToDevice);
			CudaExecute(&inputs[i * data.neuronsPerLayer[0]], &result[i * neuronsOnLastLayer], data);
			ComputeDeltas(&data, data.weightsSizeTotal / sizeof(float), data.neuronsSizeTotal / sizeof(float));

			AddCuda(data.deltasBuffer, data.deltas, data.deltasBuffer, data.neuronsSizeTotal / sizeof(float));
		}
		MultiplyScalarCuda(data.deltasBuffer, 1.0f / (*batchSize), data.neuronsSizeTotal / sizeof(float), 1, data.deltas);
		UpdateWeights(&data);
	}

	__declspec(dllexport) void CudaExecuteAndLearn(float* input, float* ideal, float* result, struct CudaData data)
	{
		cudaMemcpy(data.deviceIdeal, ideal, data.resultSizeTotal, cudaMemcpyHostToDevice);

		CudaExecute(input, result, data);

		LearnCuda(&data, data.weightsSizeTotal / sizeof(float), data.neuronsSizeTotal / sizeof(float));
	}

	__declspec(dllexport) void CudaCopy(float* output, struct CudaData data)
	{
		cudaMemcpy(output, data.deviceWeights, data.weightsSizeTotal, cudaMemcpyDeviceToHost);
	}

	__declspec(dllexport) void CudaFree(struct CudaData data)
	{
		cublasDestroy(data.handler);
		cudaFree(data.deviceIdeal);
		cudaFree(data.deviceWeights);
		cudaFree(data.deviceResult);
		cudaFree(data.layerBufferA);
		cudaFree(data.layerBufferB);
		cudaFree(data.layerBufferC);
		cudaFree(data.deltas);
		cudaFree(data.prevDeltas);
		cudaFree(data.weightBufferA);
		cudaFree(data.weightBufferB);
		cudaFree(data.deviceNeuronOutputs);
		cudaFree(data.deltasBuffer);
	}
}
