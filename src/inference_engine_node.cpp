
// std.
#include <chrono>
#include <iostream>

// OpenCV.
#include <opencv2/opencv.hpp>

// OpenVINO.
#include <inference_engine.hpp>

// ROS.
#include <ros/ros.h>

#include <sensor_msgs/Image.h>
#include <std_msgs/String.h>

// Local package.
#include <shimizu_auto_color_calib/colorir.h>

// Macros.
#define GET_PARAM(type, name, node_handle) \
    type name; \
    if ( !node_handle.getParam(#name, name) ) { \
        ROS_ERROR_STREAM("No " << #name << " parameter found. "); \
        return EXIT_FAILURE; \
    }

#define GET_PARAM_DEFAULT(type, name, default_value, node_handle) \
    type name; \
    if ( !node_handle.param<type>( #name, name, default_value ) ) { \
        ROS_WARN_STREAM("No " << #name << " parameter found. Default value " << default_value << " used. "); \
    }

using namespace InferenceEngine;

const char* DEFAULT_NODE_NAME = "colorir_pub_node";

ros::Publisher colorir_pub;

std::string input_name;
std::string output_name;
ExecutableNetwork executable_network;

InferenceEngine::Blob::Ptr wrapMat2Blob_f32(const cv::Mat &mat)
{
  size_t channels = mat.channels();
  size_t height = mat.size().height;
  size_t width = mat.size().width;

  InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::FP32,
                                    {1, channels, height, width},
                                    InferenceEngine::Layout::NHWC);

  return InferenceEngine::make_shared_blob<float>(tDesc, (float *)mat.data);
}

sensor_msgs::Image Mat2Image(cv::Mat frame_now, int channel = 3)
{

  sensor_msgs::Image output_image_msg;

  output_image_msg.height = frame_now.rows;
  output_image_msg.width = frame_now.cols;
  output_image_msg.encoding = "bgr8";
  output_image_msg.is_bigendian = false;
  output_image_msg.step = frame_now.cols * channel;
  size_t size = output_image_msg.step * frame_now.rows;
  output_image_msg.data.resize(size);
  memcpy((char *)(&output_image_msg.data[0]), frame_now.data, size);

  return output_image_msg;
}

void imageCallback(const sensor_msgs::Image::ConstPtr &image_msg)
{
  //cv::Mat color_mat(image_msg->height,image_msg->width,CV_MAKETYPE(CV_8U,3),const_cast<uchar*>(&image_msg->data[0]), image_msg->step);
  //cv::cvtColor(color_mat,color_mat,cv::COLOR_BGR2RGB);
  // ROS_INFO("I heard a image %dx%d", image_msg->width, image_msg->height);
  cv::Mat bayer_mat(image_msg->height, image_msg->width, CV_8U, const_cast<uchar *>(&image_msg->data[0]), image_msg->step);
  cv::Mat color_mat;

  cv::cvtColor(bayer_mat, color_mat, cv::COLOR_BayerRG2BGR);
  cv::resize(color_mat, color_mat, {color_mat.cols / 2, color_mat.rows / 2});

  InferRequest infer_request = executable_network.CreateInferRequest();
  cv::Mat image_f32;
  color_mat.convertTo(image_f32, CV_32FC3, 1.0 / 255);
  Blob::Ptr imgBlob = wrapMat2Blob_f32(image_f32); // just wrap Mat data by Blob::Ptr without allocating of new memory
  infer_request.SetBlob(input_name, imgBlob);      // infer_request accepts input blob of any size

  auto start = std::chrono::steady_clock::now();
  //            std::cout<<clock()*1.0/CLOCKS_PER_SEC<<"\n";
  infer_request.Infer();
  //            std::cout<<clock()*1.0/CLOCKS_PER_SEC<<"\n";
  auto duration = std::chrono::steady_clock::now() - start;
  std::cout << "Inference time = " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << "ms\n";

  Blob::Ptr output = infer_request.GetBlob(output_name);
  auto raw_ptr = output->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
  auto outbox = cv::Mat(480, 640, CV_32F, raw_ptr);
  // cv::imwrite("outbox.png",outbox*255);
  auto outptr = cv::Mat(480, 640, CV_32F, raw_ptr + 640 * 480);
  //  cv::imwrite("outptr.png",outptr*255);

  cv::Mat outbox_8u;
  outbox.convertTo(outbox_8u, CV_8UC1, 255);

  cv::Mat outptr_8u;
  outptr.convertTo(outptr_8u, CV_8UC1, 255);

  shimizu_auto_color_calib::colorir ir;

  ir.raw = Mat2Image(color_mat);
  ir.block = Mat2Image(outbox_8u, 1);
  ir.point = Mat2Image(outptr_8u, 1);

  colorir_pub.publish(ir);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, DEFAULT_NODE_NAME);

  // Node handler.
  ros::NodeHandle nh("~");

  const auto node_name = ros::this_node::getName();
  ROS_INFO_STREAM(node_name << ": Created. ");

  // ROS paramters.
  GET_PARAM_DEFAULT(std::string, topic_color_ir, "/color_ir_inference", nh)
  GET_PARAM_DEFAULT(std::string, topic_in_image, "/xic_stereo/left/image_raw", nh)
  GET_PARAM_DEFAULT(std::string, input_model, "/ws/src/shimizu_auto_color_calib/deploy/card/card.xml", nh)

  // Publisher and subscriber.
  colorir_pub = nh.advertise<shimizu_auto_color_calib::colorir>(topic_color_ir, 2);
  ros::Subscriber sub = nh.subscribe(topic_in_image, 1, imageCallback);

  // OpenVINO.
  Core ie;
  CNNNetwork network = ie.ReadNetwork(input_model);

  InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
  input_name = network.getInputsInfo().begin()->first;

  input_info->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
  input_info->setLayout(Layout::NHWC);
  input_info->setPrecision(Precision::FP32);

  DataPtr output_info = network.getOutputsInfo().begin()->second;
  output_name = network.getOutputsInfo().begin()->first;

  output_info->setPrecision(Precision::FP32);

  std::string device_name = "CPU";
  executable_network = ie.LoadNetwork(network, device_name);

  // Wait until Ctrl-C.
  ros::spin();

  return 0;
}
