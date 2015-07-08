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
#include "package_bgs/StaticFrameDifferenceBGS.h"
#include "package_bgs/WeightedMovingMeanBGS.h"
#include "package_bgs/WeightedMovingVarianceBGS.h"
#include "package_bgs/MixtureOfGaussianV1BGS.h"
#include "package_bgs/MixtureOfGaussianV2BGS.h"
#include "package_bgs/AdaptiveBackgroundLearning.h"
#include "package_bgs/AdaptiveSelectiveBackgroundLearning.h"

#if CV_MAJOR_VERSION >= 2 && CV_MINOR_VERSION >= 4 && CV_SUBMINOR_VERSION >= 3
#include "package_bgs/GMG.h"
#endif

#include "package_bgs/dp/DPAdaptiveMedianBGS.h"
#include "package_bgs/dp/DPGrimsonGMMBGS.h"
#include "package_bgs/dp/DPZivkovicAGMMBGS.h"
#include "package_bgs/dp/DPMeanBGS.h"
#include "package_bgs/dp/DPWrenGABGS.h"
#include "package_bgs/dp/DPPratiMediodBGS.h"
#include "package_bgs/dp/DPEigenbackgroundBGS.h"
#include "package_bgs/dp/DPTextureBGS.h"
#include "package_bgs/tb/T2FGMM_UM.h"
#include "package_bgs/tb/T2FGMM_UV.h"
#include "package_bgs/tb/T2FMRF_UM.h"
#include "package_bgs/tb/T2FMRF_UV.h"
#include "package_bgs/tb/FuzzySugenoIntegral.h"
#include "package_bgs/tb/FuzzyChoquetIntegral.h"
#include "package_bgs/lb/LBSimpleGaussian.h"
#include "package_bgs/lb/LBFuzzyGaussian.h"
#include "package_bgs/lb/LBMixtureOfGaussians.h"
#include "package_bgs/lb/LBAdaptiveSOM.h"
#include "package_bgs/lb/LBFuzzyAdaptiveSOM.h"
#include "package_bgs/ck/LbpMrf.h"
#include "package_bgs/jmo/MultiLayerBGS.h"
// The PBAS algorithm was removed from BGSLibrary because it is
// based on patented algorithm ViBE
// http://www2.ulg.ac.be/telecom/research/vibe/
//#include "package_bgs/pt/PixelBasedAdaptiveSegmenter.h"
#include "package_bgs/av/VuMeter.h"
#include "package_bgs/ae/KDE.h"
#include "package_bgs/db/IndependentMultimodalBGS.h"
#include "package_bgs/sjn/SJN_MultiCueBGS.h"
#include "package_bgs/bl/SigmaDeltaBGS.h"

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <image_transport/transport_hints.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>

#include <unistd.h>

#define NODE_NAME  "bgsnode"

// MixtureOfGaussianV2BGS
// AdaptiveBackgroundLearning
// AdaptiveSelectiveBackgroundLearning ***
// DPGrimsonGMMBGS
// DPZivkovicAGMMBGS
// DPEigenbackgroundBGS ***
// T2FGMM_UV
// FuzzySugenoIntegral
// FuzzyChoquetIntegral
// MultiLayerBGS
// IndependentMultimodalBGS
// SigmaDeltaBGS


class BackgroundModeller
{
public:
  BackgroundModeller(void)
   : m_imgTransport(m_nodeHandle),
     m_bgsPackage(NULL)
  {
    m_bgsPackage = new AdaptiveSelectiveBackgroundLearning;

    m_imgSubscriber = m_imgTransport.subscribe("image_raw", 1,
      &BackgroundModeller::ReceiveImage, this, image_transport::TransportHints("compressed"));

    m_imgPublisherForeground = m_imgTransport.advertise("foreground", 1);
    m_imgPublisherBackground = m_imgTransport.advertise("background", 1);
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

    if (!img_mask.empty())
    {
      if (!img_bkgmodel.empty())
      {
        img_bkgmodel.copyTo(cv_ptr->image);
        m_imgPublisherBackground.publish(cv_ptr->toImageMsg());
      }

      if (!img_mask.empty())
      {
        img_mask.copyTo(cv_ptr->image);
        cv_ptr->encoding = "mono8";
        m_imgPublisherForeground.publish(cv_ptr->toImageMsg());
      }
    }
  }

private:
  ros::NodeHandle m_nodeHandle;
  image_transport::ImageTransport m_imgTransport;
  image_transport::Subscriber m_imgSubscriber;
  image_transport::Publisher m_imgPublisherForeground;
  image_transport::Publisher m_imgPublisherBackground;
  IBGS* m_bgsPackage;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, NODE_NAME);

  if (argc >= 2)
    chdir(argv[1]);

  BackgroundModeller BackgroundModeller;
  ros::spin();
  return 0;
}
