/*
 * Copyright (c) 2014 Garrett Brown
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *   The above copyright notice and this permission notice shall be included in
 *   all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "package_bgs/FrameDifferenceBGS.h"

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <image_transport/transport_hints.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>

#define NODE_NAME  "bgsnode"

class BackgroundModeller
{
public:
  BackgroundModeller(void)
   : m_imgTransport(m_nodeHandle),
     m_bgsPackage(new FrameDifferenceBGS)
  {
    m_imgSubscriber = m_imgTransport.subscribe("/v4l/camera/image_raw", 1,
      &BackgroundModeller::ReceiveImage, this, image_transport::TransportHints("compressed"));

    m_imgPublisher = m_imgTransport.advertise("/image_processor/output_video", 1);
  }

  ~BackgroundModeller(void)
  {
    delete m_bgsPackage;
  }

  void ReceiveImage(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    cv::Mat img_mask;
    cv::Mat img_bkgmodel;
    m_bgsPackage->process(cv_ptr->image, img_mask, img_bkgmodel);

    // Draw an example circle on the video stream
    if (cv_ptr->image.rows > 60 && cv_ptr->image.cols > 60)
      cv::circle(cv_ptr->image, cv::Point(50, 50), 10, CV_RGB(255 ,0, 0));

    // Output modified video stream
    m_imgPublisher.publish(cv_ptr->toImageMsg());
  }

private:
  ros::NodeHandle m_nodeHandle;
  image_transport::ImageTransport m_imgTransport;
  image_transport::Subscriber m_imgSubscriber;
  image_transport::Publisher m_imgPublisher;
  IBGS* m_bgsPackage;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, NODE_NAME);
  BackgroundModeller BackgroundModeller;
  ros::spin();
  return 0;
}
