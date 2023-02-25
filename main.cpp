// @Author: Matheus Teixeira de Sousa (mtsousa14@gmail.com)
//
// Implement the code to detect forks with YOLOv7-fork ONNX model

#include <iostream>
#include <fstream>

// OpenCV files
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// ONNXRunTime files
#include <onnxruntime/onnxruntime_cxx_api.h>

// Model file
#include "model/yolov7.cpp"

int main(int argc, char** argv){

	std::cout << "----------------------" << std::endl;
	std::cout << "YOLOv7-fork Object Detection\n" << std::endl;

	// Check the number of arguments
	if (argc < 3 || argc > 4){
		std::cout << "Error: Expected 2 (or 3) arguments, but got " << argc << "." << std::endl;
		std::cout << "Usage: ./main model_path img_path <optional>img_output_path" << std::endl;
		std::cout << "Examples:\n";
		std::cout << "- ./main fork_best_640x640.onnx fork1.jpg\n";
		std::cout << "- ./main fork_best_640x640.onnx fork1.jpg fork1_pred.jpg\n" << std::endl;
		return -1;
	}

	// Load YOLOv7-fork model
	std::cout << "Loading model: \"" << argv[1] << "\"..."<< std::endl;
	YOLOv7 yolov7_fork(argv[1]);

	// Load and prepare image to predict
	std::cout << "Loading image: \"" << argv[2] << "\"..."<< std::endl;
	cv::Mat src_img = cv::imread(argv[2]);
	cv::Mat blob_img = yolov7_fork.image_preprocess(src_img);
	
	// Predict bounding box from image
	std::cout << "Predicting boxes..." << std::endl;
	std::vector<BoxPred> boxes = yolov7_fork.detect(blob_img);

	// Draw bounding box on image
	std::cout << "Drawing boxes...\n" << std::endl;
	cv::Scalar color = cv::Scalar(104, 184, 82); // Green color
	cv::Mat dst_img = yolov7_fork.draw_boxes(src_img, boxes, color, false, 0.0);

	// Save predicted image on output file
	if (argc == 4){
		std::cout << "Saving image...\n" << std::endl;
		cv::imwrite(argv[3], dst_img);
	}

	// Show predicted image
	std::cout << "Plotting predicted image...\n" << std::endl;
	std::string kWinName = "Predicted image";
	cv::imshow(kWinName, dst_img);
	cv::waitKey(0);
	cv::destroyAllWindows();
}
