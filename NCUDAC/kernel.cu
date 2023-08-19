
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cblas.h>
#include <omp.h>
#include <math.h>

int GetMax(int* array, int size)
{
	int max = 0;
	for (int i = 0; i < size; i++)
		if (array[i] > max)
			max = array[i];
	return max;
}

int TotalWeightCount(int* neurons, int* layerCount)
{
	int count = 0;
	for (int i = 0; i < (*layerCount - 1); i++)
		count += (neurons[i] + 1) * neurons[i + 1];
	return count;
}

extern "C"
{
	float Activate(float value)
	{
		return	1.0 / (1.0 + pow(2.718281828, -value));
	}

	void ActivateVector(float* vector, int size)
	{
		for (int i = 0; i < size; i++)
			vector[i] = Activate(vector[i]);
	}

	void ActivateToVector(float* vector, float* result, int size)
	{
		for (int i = 0; i < size; i++)
			result[i] = Activate(vector[i]);
	}

	void MultiplyBlas(int rows, int columns, float* matrix, float* vector, float* result)
	{


		cblas_sgemv(CblasRowMajor, CblasNoTrans, rows, columns, 1, matrix, columns, vector, 1, 0, result, 1);


	}

	void CopyArray(float* a, int size, float* b)
	{
		memcpy(b, a, size * 4);
		for (int i = 0; i < size; i++)
			b[i] = a[i];
	}



	/*
	* input массив входных данных
	* inputSize размер массива
	* weights массив всех весов в сети
	* weightsSizes размер массива весов
	* neurons количество нейронов на слоях
	* layerCount количество слоёв
	* result результат работы сети
	*/
	__declspec(dllexport) void BlasExecutionBias(float* input, int* inputSize, float* weights,
		int* neurons, int* layerCount, float* result)
	{

		//Позиция в массиве весов
		int weightPosition = 0;

		//Максимальный размер слоя, с учётом нейронов смещения (+1)
		int maxVectorSize = GetMax(neurons, *layerCount) + 1;

		//Выделяем два буфера для вычислений
		float* nextLayerInput = (float*)malloc(maxVectorSize * sizeof(float));	//Значения входных нейронов 
		float* output = (float*)malloc(maxVectorSize * sizeof(float));			//Значения выходных нейронов


		//Копируем входные данные сети в отдельные массив
		CopyArray(input, *inputSize, nextLayerInput);
		nextLayerInput[neurons[0]] = 1; //Установка нейрона смещения

		for (int layer = 0; layer < (*layerCount - 2); layer++)
		{
			/* Умножаем матрицу весов на вектор входных нейронов.
			Количество столбцов в матрице равно количеству входных нейронов.
			С учётом нейронов смещения это neurons[layer]+1 */
			MultiplyBlas(neurons[layer + 1], neurons[layer] + 1, &weights[weightPosition], nextLayerInput, output);

			/* Обновляем нашу текущую позицию в массиве весов.
			Смещение на каждом шаге в этом массиве вычисляется как произведение количества нейронов
			на текущем слое и на следующем. С учётом нейрона смещения в текущем слое, это
			(neurons[layer]+1) * neurons[layer + 1
			*/

			weightPosition += (neurons[layer] + 1) * neurons[layer + 1];

			/* Применяем функцию активации к массиву данных и копируем эти
			значения в массив входных данных следующего слоя */
			ActivateToVector(output, nextLayerInput, neurons[layer + 1]);

			//Устанавливаем нейрон смещения, добавлением в конец вектора нейронов единицы
			nextLayerInput[neurons[layer + 1]] = 1;
		}

		//Вычисления последнего слоя сохраняем в входной массив
		MultiplyBlas(neurons[*layerCount - 1], neurons[*layerCount - 2] + 1, &weights[weightPosition], nextLayerInput, result);

		//Применяем функцию активации к выходу сети
		ActivateVector(result, neurons[*layerCount - 1]);

		//Освобождаем выделенную память
		free(nextLayerInput);
		free(output);
	}

	__declspec(dllexport) void BlasExecutionBiasMany(float* input, int* singleInputSize, int* inputCount, float* weights,
		int* neurons, int* layerCount, float* result)
	{
		int outputSize = neurons[*layerCount - 1];
		for (int i = 0; i < *inputCount; i++)
		{
			BlasExecutionBias(&input[i * (*singleInputSize)], singleInputSize, weights, neurons, layerCount, &result[outputSize * i]);
		}
	}
}
