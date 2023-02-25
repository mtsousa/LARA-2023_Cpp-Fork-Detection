// @Author: Matheus Teixeira de Sousa (mtsousa14@gmail.com)
//
// Implement the class of YOLOv7 model
//
// Based on https://github.com/hpc203/yolov7-opencv-onnxrun-cpp-py

// Bounding box struct
struct BoxPred{
	float xmin;
	float ymin;
	float xmax;
	float ymax;
	int pred_label;
	float pred_conf;
};

// YOLOv7 class
class YOLOv7{
	public:
		YOLOv7(const ORTCHAR_T* model_path);
		std::vector<BoxPred> detect(cv::Mat& blob);
		cv::Mat image_preprocess(cv::Mat& img);
		cv::Mat draw_boxes(cv::Mat& src_img, std::vector<BoxPred>& pred_box, cv::Scalar color, bool mask, double mask_alpha);

	private:
		std::array<int, 2> input_size;
		std::vector<std::string> class_names;
		float ratio_w, ratio_h;

		std::vector<float> matrix_to_vector(cv::Mat img);

		Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "YOLOv7");
		Ort::Session *ort_session = nullptr;
		Ort::SessionOptions sessionOptions = Ort::SessionOptions();
		std::vector<char*> input_names;
		std::vector<char*> output_names;
};

YOLOv7::YOLOv7(const ORTCHAR_T* model_path){
	/*
	Initialize the class
	
	Args
		- model_path: Path to ONNX model
	*/
	std::string classesFile = "coco.names";
	std::vector<std::vector<int64_t>> input_node_dims; // >=1 outputs
	this->sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Ort::Session(this->env, model_path, this->sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	Ort::AllocatorWithDefaultOptions allocator;
	
	// For each input, get the name and the shape
	for (int i = 0; i < numInputNodes; i++){
		auto tmp = ort_session->GetInputNameAllocated(i, allocator);
		input_names.push_back(new char[strlen(tmp.get()) + 1]);
		strcpy(input_names[i], tmp.get());
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	
	// For each output, get the name
	for (int i = 0; i < numOutputNodes; i++){
		auto tmp = ort_session->GetOutputNameAllocated(i, allocator);
		output_names.push_back(new char[strlen(tmp.get()) + 1]);
		strcpy(output_names[i], tmp.get());
	}

	this->input_size[0] = input_node_dims[0][2]; // Height
	this->input_size[1] = input_node_dims[0][3]; // Width

	// Get the output label names
	std::ifstream ifs(classesFile.c_str());
	std::string line;
	while (getline(ifs, line)){
		this->class_names.push_back(line);
	}

	std::cout << "Model params:" << std::endl;
	std::cout << "- Input image size (h x w): " << this->input_size[0] << " x " << this->input_size[1] << std::endl;
	std::cout << "- Number of classes: " << this->class_names.size() << "\n" << std::endl;
}

std::vector<float> YOLOv7::matrix_to_vector(cv::Mat img){
	/*
	Convert a matrix (image) to a continuous vector

	Args
		- img: Image to be transformed
	
	Returns
		- input_image: Vector of input image
	
	https://stackoverflow.com/a/56600115
	*/
	int b = img.size[0], c = img.size[1], h = img.size[2], w = img.size[3];
	cv::Mat flat = img.reshape(1, b*c*h*w);
	std::vector<float> input_image;
	input_image = img.isContinuous()? flat : flat.clone();

	return input_image;
}

cv::Mat YOLOv7::draw_boxes(cv::Mat& src_img, std::vector<BoxPred>& pred_box, cv::Scalar color, bool mask, double mask_alpha){
	/*
	Draw bounding boxes on image

	Args
		- src_img: Original image
		- pred_box: Vector of bounding boxes
		- color: Color of bounding box
		- mask: Flag to add the mask
		- mask_alpha: Visibility of the mask

	Returns
		- dst_img: Image with drawed boxes
	*/
	cv::Mat dst_img, mask_img;
	src_img.copyTo(dst_img);
	src_img.copyTo(mask_img);

	float tl = round(0.002 * (dst_img.rows + dst_img.cols) / 2) + 1;
	float tf = (tl - 1) > 1 ? (tl - 1) : 1;

	// For each prediction, draw the bounding box
	for (size_t i = 0; i < pred_box.size(); ++i){
		auto c1 = cv::Point(int(pred_box[i].xmin), int(pred_box[i].ymin));
		auto c2 = cv::Point(int(pred_box[i].xmax), int(pred_box[i].ymax));

		cv::rectangle(dst_img, c1, c2, color, tl, cv::LINE_AA);
		cv::rectangle(mask_img, c1, c2, color, -1, cv::LINE_AA); // Filled

		std::string label = cv::format("%.3f", pred_box[i].pred_conf);
		label = this->class_names[pred_box[i].pred_label] + " " + label;
		auto t_size = cv::getTextSize(label, 1, tl / 3, tf, 0);

        c2 = cv::Point(int(pred_box[i].xmin) + t_size.width, int(pred_box[i].ymin) - t_size.height - 5);
		
		cv::rectangle(dst_img, c1, c2, color, -1, cv::LINE_AA);  // Filled
		cv::putText(dst_img, label, cv::Point(int(pred_box[i].xmin), int(pred_box[i].ymin) - 2), 1, tl/3, cv::Scalar(255, 255, 255), tf, cv::LINE_AA);

		cv::rectangle(mask_img, c1, c2, color, -1, cv::LINE_AA);  // Filled
		cv::putText(mask_img, label, cv::Point(int(pred_box[i].xmin), int(pred_box[i].ymin) - 2), 1, tl/3, cv::Scalar(255, 255, 255), tf, cv::LINE_AA);
	}

	// If mask is true, add a mask
	if (mask){
		cv::Mat out_img;
		cv::addWeighted(mask_img, mask_alpha, dst_img, 1 - mask_alpha, 0.0, out_img);
		return out_img;
	}

	return dst_img;
}

std::vector<BoxPred> YOLOv7::detect(cv::Mat& blob){
	/*
	Predict the bounding boxes

	Args
		- blob: Batched image (b x c x h x w)

	Returns
		- pred_boxes: Predicted bounding boxes
	*/
	// Convert batch to continuous vector
	std::vector<float> input_image = this->matrix_to_vector(blob);
	std::array<int64_t, 4> input_shape {blob.size[0], blob.size[1], blob.size[2], blob.size[3]};

	// Alloc memory and create input tensor
	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, input_image.data(), input_image.size(), input_shape.data(), input_shape.size());

	// Run the inference
	std::vector<Ort::Value> ort_outputs = ort_session->Run(Ort::RunOptions{ nullptr }, &this->input_names[0], &input_tensor, 1, this->output_names.data(), this->output_names.size());
	
	int n = 0, k = 0;
	std::vector<BoxPred> pred_boxes;
	const float* pdata = ort_outputs[0].GetTensorData<float>(); // Get the predictions

	// https://stackoverflow.com/a/74574180
	// Adjust predictions ratio and save the struct
	auto outputInfo = ort_outputs[0].GetTensorTypeAndShapeInfo();
	int rows = outputInfo.GetShape()[0], cols = outputInfo.GetShape()[1];

	for (n = 0; n < rows; n++){
		k = n*cols;
		float xmin {pdata[k + 1]*this->ratio_w}, ymin {pdata[k + 2]*this->ratio_h}, xmax {pdata[k + 3]*this->ratio_w}, ymax {pdata[k + 4]*this->ratio_h}, pred_class {pdata[k + 5]},conf {pdata[k + 6]};
		pred_boxes.push_back(BoxPred{ xmin, ymin, xmax, ymax, int(pred_class), conf });
	}

	std::cout << "- Predicted boxes: " << rows << "\n" << std::endl;
	
	return pred_boxes;
}

cv::Mat YOLOv7::image_preprocess(cv::Mat& img){
	/*
	Adjust image shape and normalize
	
	Args
		- img: Original image

	Returns
		- batched_img: Batched image (b x c x h x w)
	*/
    cv::Mat adj_img, batched_img;
	cv::Size model_size = cv::Size(this->input_size[1], this->input_size[0]), img_size = img.size(); // w x h

	// Compute ratio for image scale
	this->ratio_h = (float)img.rows/this->input_size[0];
	this->ratio_w = (float)img.cols/this->input_size[1];

	// If needed, resize the image
	if (model_size != img_size){
		cv::resize(img, adj_img, model_size, 0, 0, cv::INTER_LINEAR);
	} else{
		img.copyTo(adj_img);
	}

	std::cout << "Image params:" << std::endl;
	std::cout << "- Original size (h x w): " << img.size << std::endl;
	std::cout << "- Adjusted size (h x w): " << adj_img.size << std::endl;
    
	// Expand the image to (b, c, h, w) and normalize 
	cv::dnn::blobFromImage(adj_img, batched_img, 1 / 255.0f, cv::Size(adj_img.cols, adj_img.rows), cv::Scalar(0, 0, 0), true, false);
	std::cout << "- Batched size (b x c x h x w): " << batched_img.size << "\n" << std::endl;

	return batched_img;
}