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


vector<string> loadBatchedFileList(string path, int batchSize)
{
	vector<string> _fileList;
	_fileList.clear();
	
	for (auto &entry : std::experimental::filesystem::directory_iterator(path))
	{
		_fileList.push_back(entry.path().string());
	}
	int batch_num = int(_fileList.size() / batchSize);
	int modulo = _fileList.size() % batchSize;

	if (!(modulo == 0))
	{
		cout << "File size is " << _fileList.size() << "... adding dummy inputs at the end of file list..." << endl;
		
		int count = 0;
		while (count < batchSize - modulo)
		{
			_fileList.push_back(_fileList.back());
			count += 1;
		}
		cout << "Now the file size is " << _fileList.size() << endl;
		batch_num += 1;
	}
	else
	{
		cout << "File size is " << _fileList.size() << ", the multiple of " << batchSize << "." << endl;
	}

	return _fileList;
}


/* 
Load images with arbitrary batch number. If the number of image does not divide by batch number,
this function adds dummy inputs (last file) in order to fix the data structure.
*/
bool loadBatchedInput(string folderName, int batchSize, vector<cv::Mat> &cv2_imgList, int &w, int &h)
{
	vector<string> fileList;
	fileList = loadBatchedFileList(folderName, batchSize);

	if (!cv2_imgList.empty()) cv2_imgList.clear();
	cv2_imgList.resize(fileList.size());

	for (int i = 0; i < fileList.size(); i++) {
			cv2_imgList[i] = imread(fileList[i].c_str(), 0);
			cv2_imgList[i].convertTo(cv2_imgList[i], CV_32FC1);
			if (!(cv2_imgList[0].cols == cv2_imgList[i].cols && cv2_imgList[0].rows == cv2_imgList[i].rows))
			{
				cout << "All images should have same size." << endl;
				return false;
			}
	}
	w = cv2_imgList[0].cols;
	h = cv2_imgList[0].rows;
	return true;
}

//
bool saveLabels_array(float *src, int w, int h, string path, int batchSize)
{
	if (src == nullptr) return false;
	
	char b2[200];
	vector<string> fileList;
	fileList = loadBatchedFileList(path, batchSize);
	for (int i = 0; i < fileList.size(); i++)
	{
		fileList[i] = regex_replace(fileList[i], regex("\.bmp$|\.png$|\.pgm$"), "\.txt");
		fileList[i] = regex_replace(fileList[i], regex("input"), "output");

		ofstream writefile;
		writefile.open(fileList[i]);
		if (writefile.is_open()) {
			for (int m = 0; m < h; m++) {
				for (int n = 0; n < w; n++) {
					writefile << static_cast<int>(src[i*w*h + m*w + n]);
					writefile << "\t";
				}
				writefile << "\n";
			}
		}
		writefile.close();
	}
	return true;
}