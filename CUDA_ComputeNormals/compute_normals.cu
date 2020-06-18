#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cutil_math.h"
#include <stdio.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <chrono>
using namespace std;

__device__ float3 PlaneFit(float3 points[10], int nNeighbours)
{
	if (nNeighbours < 3)
		return float3();

	float3 sum = float3();
	for (int i = 0; i < nNeighbours; i++)
		sum += points[i];
	float3 centro = sum * (1.0f / nNeighbours);

	// Calc full 3x3 covariance matrix, excluding symmetries:
	float xx = 0.0f;
	float xy = 0.0f;
	float xz = 0.0f;
	float yy = 0.0f;
	float yz = 0.0f;
	float zz = 0.0f;

	for (int i = 0; i < nNeighbours; i++)
	{
		float3 r = (points[i] - centro);
		xx += r.x * r.x;
		xy += r.x * r.y;
		xz += r.x * r.z;
		yy += r.y * r.y;
		yz += r.y * r.z;
		zz += r.z * r.z;
	}

	float det_x = yy * zz - yz * yz;
	float det_y = xx * zz - xz * xz;
	float det_z = xx * yy - xy * xy;

	float det_max = max(max(det_x, det_y), det_z);

	if (det_max <= 0.0f)
		return float3(); // The points don't span a plane

	// Pick path with best conditioning:
	float3 dir = float3();

	if (det_max == det_x) {
		dir = float3();
		dir.x = det_x;
		dir.y = xz * yz - xy * zz;
		dir.z = xy * yz - xz * yy;
	}
	else if (det_max == det_y) {
		dir = float3();
		dir.x = xz * yz - xy * zz;
		dir.y = det_y;
		dir.z = xy * xz - yz * xx;
	}
	else {
		dir = float3();
		dir.x = xy * yz - xz * yy;
		dir.y = xy * xz - yz * xx;
		dir.z = det_z;
	}

	return normalize(dir);
}

__global__ void ComputeNormals(float3* points, float3* normals, int nRows, int nColumns, int nNearCells, float3 refDirection)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int column = blockIdx.y * blockDim.y + threadIdx.y;
	if (row >= nRows || column >= nColumns)return;

	int idx = row * nColumns + column; //current index

	int lbr = max(0, row - nNearCells);
	int hbr = min(nRows, row + nNearCells);
	int lbc = max(0, column - nNearCells);
	int hbc = min(nColumns, column + nNearCells);

	float bestDist[10];
	float3 bestPoints[10];
	for (int i = 0; i < 10; i++)
		bestDist[i] = 2e20f;

	int found = 0;

	for (int r = lbr; r < hbr; r++)
	{
		for (int c = lbc; c < hbc; c++)
		{
			// Considering neighbor only if valid and within max distance
			int neighborIdx = r * nColumns + c;

			float dist = distance(points[idx], points[neighborIdx]);
			found++;
			int i = 9;
			while (i > 0 && dist < bestDist[i - 1])
			{
				bestDist[i] = bestDist[i - 1];
				bestPoints[i] = bestPoints[i - 1];
				i--;
			}
			if (i < 9)
			{
				bestDist[i] = dist;
				bestPoints[i] = points[neighborIdx];
			}
		}
	}

	// Compute normals using least squares estimation
	float3 normal = PlaneFit(bestPoints, min(found, 10));

	// Re-orient normal correctly using user reference normals direction
	if (dot(normal, refDirection) > 0.0f)
		normals[idx] = normal;
	else
		normals[idx] = normal * -1; //flip normal
}


int main()
{
	float3* points = NULL;

	string filename = "C:\\Users\\Alessandro\\Desktop\\CUDAComputeNormals\\Gear.pcd";

#pragma region Load from file

	int width, height, nPoints;

	FILE* infile = fopen(filename.c_str(), "r");

	if (infile == NULL) {
		cerr << "Failed to load file!" << endl;
		return;
	}

	char line[100];
	for (int i = 0; i < 6; i++)
		fscanf(infile, "%*[^\n]\n");

	//Width, height
	fscanf(infile, "%*s %i\n", &width);
	fscanf(infile, "%*s %i\n", &height);

	fscanf(infile, "%*[^\n]\n"); //skip one line

	//Points count
	fscanf(infile, "%*s %i\n", &nPoints);

	fscanf(infile, "%*[^\n]\n"); //skip one line

	points = (float3*)malloc(nPoints * sizeof(float3));

	int i = 0;
	while (true)
	{
		float x = 0, y = 0, z = 0, rgb = 0, nx = 0, ny = 0, nz = 0;
		if (fscanf(infile, "%f %f %f %f %f %f %f\n", &x, &y, &z, &rgb, &nx, &ny, &nz) == EOF) { break; }
		points[i].x = x;
		points[i].y = y;
		points[i].z = z;
		i++;
	}

	fclose(infile);

#pragma endregion

	float3* normals = (float3*)malloc(nPoints * sizeof(float3));

	float3* gPoints;
	cudaMalloc(&gPoints, nPoints * sizeof(float3));
	float3* gNormals;
	cudaMalloc(&gNormals, nPoints * sizeof(float3));

	//Load input points
	cudaMemcpy(gPoints, points, nPoints * sizeof(float3), cudaMemcpyHostToDevice);

	//Run kernel
	float3 refDirection = float3();
	refDirection.x = 0;
	refDirection.y = 0;
	refDirection.z = -1;

	dim3 dimBlock(32, 32); // so your threads are BLOCK_SIZE*BLOCK_SIZE, 256 in this case
	dim3 dimGrid(ceil(height / 32), ceil(width / 32)); // 1*1 blocks in a grid

	auto start = chrono::steady_clock::now();
	ComputeNormals << <dimGrid, dimBlock >> > (gPoints, gNormals, height, width, 3, refDirection);
	//Get output normals
	cudaMemcpy(normals, gNormals, nPoints * sizeof(float3), cudaMemcpyDeviceToHost);
	auto end = chrono::steady_clock::now();

	cout << "Elapsed time in milliseconds (CUDA): "
		<< chrono::duration_cast<chrono::milliseconds>(end - start).count()
		<< " ms" << endl;

	filename = "C:\\Users\\Alessandro\\Desktop\\CUDAComputeNormals\\GearWithNormals.pcd";

#pragma region Save to file

	FILE* outfile = fopen(filename.c_str(), "w+");

	string header = "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z normal_x normal_y normal_z\nSIZE 4 4 4 4 4 4\nTYPE F F F F F F\nCOUNT 1 1 1 1 1 1\nWIDTH " + to_string(width) + "\nHEIGHT " + to_string(height) + "\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS " + to_string(nPoints) + "\nDATA ascii\n";

	if (outfile != NULL)
	{
		fprintf(outfile, "%s", header.c_str());

		for (int i = 0; i < nPoints; i++)
		{
			fprintf(outfile, "%f %f %f %f %f %f\n", points[i].x, points[i].y, points[i].z, normals[i].x, normals[i].y, normals[i].z);
		}
		fclose(outfile);
	}
	else
	{
		cerr << "Unable to save file" << endl;
		perror("Error");
	}

#pragma endregion

	cudaFree(gNormals);
	cudaFree(gPoints);
	free(normals);
	free(points);
}
