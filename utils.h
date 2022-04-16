#pragma once
#include <fstream>
#include <iostream>
#include <string>
#include <regex>
#include <vector>
#include <algorithm>
#include <math.h>
#include <filesystem>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

bool loadBatchedInput(string folderName, int batchSize, vector<cv::Mat> &cv2_imgList, int &w, int &h);
vector<string> loadBatchedFileList(string path, int batchSize);
bool saveLabels_array(float *src, int w, int h, string path, int batchSize);