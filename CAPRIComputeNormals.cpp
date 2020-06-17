#include <iostream>
#include <omp.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
using namespace std;

class vector3
{
public:
	float x, y, z;

	vector3(float xx, float yy, float zz)
	{
		x = xx;
		y = yy;
		z = zz;
	}

	vector3()
	{
		x = 0;
		y = 0;
		z = 0;
	}

	static float dot(vector3& v1, vector3& v2)
	{
		return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
	}

	static float distance(vector3& v1, vector3& v2)
	{
		return sqrt(pow(v2.x - v1.x, 2) + pow(v2.y - v1.y, 2) + pow(v2.z - v1.z, 2));
	}

	vector3 normalize()
	{
		return *this / sqrt(dot(*this, *this));
	}

	vector3 operator+(vector3& v2)
	{
		vector3 ret = vector3(0, 0, 0);
		ret.x = this->x + v2.x;
		ret.y = this->y + v2.y;
		ret.z = this->z + v2.z;

		return ret;
	}

	vector3 operator-(vector3& v2)
	{
		vector3 ret = vector3(0, 0, 0);
		ret.x = this->x - v2.x;
		ret.y = this->y - v2.y;
		ret.z = this->z - v2.z;

		return ret;
	}

	vector3 operator*(float f)
	{
		vector3 ret = vector3(0, 0, 0);
		ret.x = this->x * f;
		ret.y = this->y * f;
		ret.z = this->z * f;

		return ret;
	}

	vector3 operator/ (float f)
	{
		vector3 ret = vector3(0, 0, 0);
		ret.x = this->x / f;
		ret.y = this->y / f;
		ret.z = this->z / f;

		return ret;
	}

	vector3 operator+=(vector3& v2)
	{
		this->x = this->x + v2.x;
		this->y = this->y + v2.y;
		this->z = this->z + v2.z;

		return *this;
	}
};

vector3 PlaneFit(vector3 points[10], int nNeighbours)
{
	if (nNeighbours < 3)
		return vector3();

	vector3 sum = vector3();
	for (int i = 0; i < nNeighbours; i++)
		sum += points[i];
	vector3 centro = sum * (1.0f / nNeighbours);

	// Calc full 3x3 covariance matrix, excluding symmetries:
	float xx = 0.0f;
	float xy = 0.0f;
	float xz = 0.0f;
	float yy = 0.0f;
	float yz = 0.0f;
	float zz = 0.0f;

	for (int i = 0; i < nNeighbours; i++)
	{
		vector3 r = (points[i] - centro);
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
		return vector3(); // The points don't span a plane

	// Pick path with best conditioning:
	vector3 dir = vector3();

	if (det_max == det_x)
		dir = vector3(det_x, xz * yz - xy * zz, xy * yz - xz * yy);
	else if (det_max == det_y)
		dir = vector3(xz * yz - xy * zz, det_y, xy * xz - yz * xx);
	else
		dir = vector3(xy * yz - xz * yy, xy * xz - yz * xx, det_z);

	return dir.normalize();
}

void ComputeNormals(vector<vector3>& points, vector<vector3>& normals, int nRows, int nColumns, int nNearCells, vector3& refDirection)
{
	for (int row = 0; row < nRows; row++)
	{
		//cout << "Sono il thread: " << omp_get_thread_num() << endl;
		for (int column = 0; column < nColumns; column++)
		{
			int idx = row * nColumns + column; //current index

			int lbr = max(0, row - nNearCells);
			int hbr = min(nRows, row + nNearCells);
			int lbc = max(0, column - nNearCells);
			int hbc = min(nColumns, column + nNearCells);

			float bestDist[10];
			vector3 bestPoints[10];
			for (int i = 0; i < 10; i++)
				bestDist[i] = 2e20f;

			int found = 0;

			for (int r = lbr; r < hbr; r++)
			{
				for (int c = lbc; c < hbc; c++)
				{
					// Considering neighbor only if valid and within max distance
					int neighborIdx = r * nColumns + c;

					float dist = vector3::distance(points[idx], points[neighborIdx]);
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
			vector3 normal = PlaneFit(bestPoints, min(found, 10));

			// Re-orient normal correctly using user reference normals direction
			if (vector3::dot(normal, refDirection) > 0.0f)
				normals[idx] = normal;
			else
				normals[idx] = normal * -1; //flip normal
		}
	}
}

void ParallelComputeNormals(vector<vector3>& points, vector<vector3>& normals, int nRows, int nColumns, int nNearCells, vector3& refDirection)
{
#pragma omp parallel for
	for (int row = 0; row < nRows; row++)
	{
		//cout << "Sono il thread: " << omp_get_thread_num() << endl;
#pragma omp parallel for
		for (int column = 0; column < nColumns; column++)
		{
			int idx = row * nColumns + column; //current index

			int lbr = max(0, row - nNearCells);
			int hbr = min(nRows, row + nNearCells);
			int lbc = max(0, column - nNearCells);
			int hbc = min(nColumns, column + nNearCells);

			float bestDist[10];
			vector3 bestPoints[10];
			for (int i = 0; i < 10; i++)
				bestDist[i] = 2e20f;

			int found = 0;

			for (int r = lbr; r < hbr; r++)
			{
				for (int c = lbc; c < hbc; c++)
				{
					// Considering neighbor only if valid and within max distance
					int neighborIdx = r * nColumns + c;

					float dist = vector3::distance(points[idx], points[neighborIdx]);
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
			vector3 normal = PlaneFit(bestPoints, min(found, 10));

			// Re-orient normal correctly using user reference normals direction
			if (vector3::dot(normal, refDirection) > 0.0f)
				normals[idx] = normal;
			else
				normals[idx] = normal * -1; //flip normal
		}
	}
}

void LoadFromPCD(string filename, vector<vector3>& outputVec, int& width, int& height)
{
	ifstream infile(filename);

	string line;
	for (int i = 0; i < 6; i++)
		getline(infile, line);

	//Width, height
	getline(infile, line);
	width = stoi(line.substr(6));
	getline(infile, line);
	height = stoi(line.substr(7));

	getline(infile, line); //skip one line

	//Points count
	getline(infile, line);
	int nPoints = stoi(line.substr(7));

	outputVec = vector<vector3>(nPoints);
	getline(infile, line); //skip one line

	int i = 0;
	while (getline(infile, line))
	{
		std::istringstream iss(line);
		float a, b, c, d, e, f, g;
		iss >> a >> b >> c;
		outputVec[i].x = a;
		outputVec[i].y = b;
		outputVec[i].z = c;
		i++;
	}
}

void FasterLoadFromPCD(string filename, vector<vector3>& outputVec, int& width, int& height)
{
	FILE* infile = fopen(filename.c_str(), "r");

	if (infile == NULL) {
		cout << "Failed to open file!";
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
	int nPoints = 0;
	fscanf(infile, "%*s %i\n", &nPoints);

	fscanf(infile, "%*[^\n]\n"); //skip one line

	outputVec = vector<vector3>(nPoints);
	int i = 0;
	while (true)
	{
		float x = 0, y = 0, z = 0, rgb = 0, nx = 0, ny = 0, nz = 0;
		if (fscanf(infile, "%f %f %f %f %f %f %f\n", &x, &y, &z, &rgb, &nx, &ny, &nz) == EOF) { break; }
		outputVec[i].x = x;
		outputVec[i].y = y;
		outputVec[i].z = z;
		i++;
	}

	fclose(infile);
}

void SaveToPCD(string filename, vector<vector3>& points, vector<vector3>& normals, int width, int height, int nPoints)
{
	ofstream outfile(filename);

	string header = "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z normal_x normal_y normal_z\nSIZE 4 4 4 4 4 4\nTYPE F F F F F F\nCOUNT 1 1 1 1 1 1\nWIDTH " + to_string(width) + "\nHEIGHT " + to_string(height) + "\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS " + to_string(nPoints) + "\nDATA ascii\n";

	if (outfile.is_open())
	{
		outfile << header;

		for (int i = 0; i < points.size(); i++)
		{
			outfile << points[i].x << " " << points[i].y << " " << points[i].z << " " << normals[i].x << " " << normals[i].y << " " << normals[i].z << endl;
		}

		outfile.close();
	}
	else cerr << "Unable to open file";

}

void FasterSaveToPCD(string filename, vector<vector3>& points, vector<vector3>& normals, int width, int height, int nPoints)
{
	FILE* outfile = fopen(filename.c_str(), "w+");

	string header = "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z normal_x normal_y normal_z\nSIZE 4 4 4 4 4 4\nTYPE F F F F F F\nCOUNT 1 1 1 1 1 1\nWIDTH " + to_string(width) + "\nHEIGHT " + to_string(height) + "\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS " + to_string(nPoints) + "\nDATA ascii\n";

	if (outfile != NULL)
	{
		fprintf(outfile, "%s", header.c_str());

		for (int i = 0; i < points.size(); i++)
		{
			fprintf(outfile, "%f %f %f %f %f %f\n", points[i].x, points[i].y, points[i].z, normals[i].x, normals[i].y, normals[i].z);
		}
		fclose(outfile);
	}
	else cerr << "Unable to open file";
}

int main()
{
	vector<vector3> points;
	int width, height;
	FasterLoadFromPCD("Gear.pcd", points, width, height);

	auto start = chrono::steady_clock::now();
	vector<vector3> normals(points.size());
	vector3 refDirection = vector3(0, 0, -1);
	ComputeNormals(points, normals, height, width, 3, refDirection);
	auto end = chrono::steady_clock::now();

	cout << "Elapsed time in milliseconds: "
		<< chrono::duration_cast<chrono::milliseconds>(end - start).count()
		<< " ms" << endl;

	start = chrono::steady_clock::now();
	ParallelComputeNormals(points, normals, height, width, 3, refDirection);
	end = chrono::steady_clock::now();

	cout << "Elapsed time in milliseconds (parallel): "
		<< chrono::duration_cast<chrono::milliseconds>(end - start).count()
		<< " ms" << endl;

	FasterSaveToPCD("GearWithNormals.pcd", points, normals, width, height, points.size());
}
