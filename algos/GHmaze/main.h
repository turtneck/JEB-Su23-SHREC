#ifndef _MAIN_H
#define _MAIN_H

#include <vector>
#include <unordered_set>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Point {
public: 
	int r;
	int c;

	__host__ __device__ Point() {
		r = -1;
		c = -1;
	}

	__host__ __device__ Point(int x, int y) {
		r = x;
		c = y;
	}

	__host__ __device__ void printPoint() {
		std::cout << "r: " << this->r << " ";
		std::cout << "c: " << this->c << "\n";
	}

	__host__ __device__ int getR() {
		return this->r;
	}

	__host__ __device__ int getC() {
		return this->c;
	}

	__host__ __device__ void setR(int x) {
		r = x;
	}
};

// Data structure for stack
struct stack
{
	int maxsize;	// define max capacity of stack
	int top;
	Point* items;
};

// Utility function to initialize stack
__host__ __device__ struct stack* newStack(int capacity)
{
	struct stack* pt = (struct stack*)malloc(sizeof(struct stack));

	pt->maxsize = capacity;
	pt->top = -1;
	pt->items = (Point*)malloc(sizeof(Point) * capacity);

	return pt;
}

// Utility function to return the size of the stack
__host__ __device__ int size(struct stack* pt)
{
	return pt->top + 1;
}

// Utility function to check if the stack is empty or not
__host__ __device__ int isEmpty(struct stack* pt)
{
	return pt->top == -1;	// or return size(pt) == 0;
}

// Utility function to check if the stack is full or not
__host__ __device__ int isFull(struct stack* pt)
{
	return pt->top == pt->maxsize - 1;	// or return size(pt) == pt->maxsize;
}

// Utility function to add an element x in the stack
__host__ __device__ void push(struct stack* pt, Point x)
{
	// check if stack is already full. Then inserting an element would 
	// lead to stack overflow
	if (isFull(pt))
	{
		printf("OverFlow\nProgram Terminated\n");
		exit(EXIT_FAILURE);
	}

	printf("Inserting %d\n", x);
	pt->items[++pt->top] = x;
}

// Utility function to return top element in a stack
__host__ __device__ Point peek(struct stack* pt)
{
	// check for empty stack
	if (!isEmpty(pt))
		return pt->items[pt->top];
	else
		exit(EXIT_FAILURE);
}

// Utility function to pop top element from the stack
__host__ __device__ Point pop(struct stack* pt)
{
	// check for stack underflow
	if (isEmpty(pt))
	{
		printf("UnderFlow\nProgram Terminated\n");
		exit(EXIT_FAILURE);
	}

	printf("Removing %d\n", peek(pt));

	// decrease stack size by 1 and (optionally) return the popped element
	return pt->items[pt->top--];
}
#endif // !_MAIN_H
