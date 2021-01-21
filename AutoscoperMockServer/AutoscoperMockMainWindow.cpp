
// Based on autoscoper/src/ui/AutoscoperMainWindow.cpp

// Qt includes
#include <QDebug>

#include "AutoscoperMockMainWindow.h"

// --------------------------------------------------------------------------
AutoscoperMockMainWindow::AutoscoperMockMainWindow()
{
  this->Pose.push_back(0.1);
  this->Pose.push_back(1.2);
  this->Pose.push_back(2.3);
  this->Pose.push_back(3.4);
  this->Pose.push_back(4.5);
  this->Pose.push_back(6.7);
}

// --------------------------------------------------------------------------
void AutoscoperMockMainWindow::setFrame(int frame)
{
  qInfo() << "setFrame";
  qInfo() << " frame" << frame;
  this->Frame = frame;
}

// --------------------------------------------------------------------------
void AutoscoperMockMainWindow::openTrial(QString filename)
{
  qInfo() << "openTrial";
  qInfo() << " filename" << filename;
}

// --------------------------------------------------------------------------
void AutoscoperMockMainWindow::load_tracking_results(QString filename, bool save_as_matrix, bool save_as_rows, bool save_with_commas, bool convert_to_cm, bool convert_to_rad, bool interpolate, int volume)
{
  qInfo() << "load_tracking_results";
  qInfo() << " filename" << filename;
  qInfo() << " save_as_matrix" << save_as_matrix;
  qInfo() << " save_as_rows" << save_as_rows;
  qInfo() << " save_with_commas" << save_with_commas;
  qInfo() << " convert_to_cm" << convert_to_cm;
  qInfo() << " convert_to_rad" << convert_to_rad;
  qInfo() << " interpolate" << interpolate;
  qInfo() << " volume" << volume;
}

// --------------------------------------------------------------------------
void AutoscoperMockMainWindow::save_tracking_results(QString filename, bool save_as_matrix, bool save_as_rows, bool save_with_commas, bool convert_to_cm, bool convert_to_rad, bool interpolate, int volume)
{
  qInfo() << "save_tracking_results";
  qInfo() << " filename" << filename;
  qInfo() << " save_as_matrix" << save_as_matrix;
  qInfo() << " save_as_rows" << save_as_rows;
  qInfo() << " save_with_commas" << save_with_commas;
  qInfo() << " convert_to_cm" << convert_to_cm;
  qInfo() << " convert_to_rad" << convert_to_rad;
  qInfo() << " interpolate" << interpolate;
  qInfo() << " volume" << volume;
}

// --------------------------------------------------------------------------
void AutoscoperMockMainWindow::loadFilterSettings(int camera, QString filename)
{
  qInfo() << "loadFilterSettings";
  qInfo() << " camera" << camera;
  qInfo() << " filename" << filename;
}

// --------------------------------------------------------------------------
std::vector<double> AutoscoperMockMainWindow::getPose(unsigned int volume, unsigned int frame)
{
  qInfo() << "getPose";
  qInfo() << " volume" << volume;
  qInfo() << " frame" << frame;
  return this->Pose;
}

// --------------------------------------------------------------------------
void AutoscoperMockMainWindow::setPose(std::vector<double> pose, unsigned int volume, unsigned int frame)
{
  qInfo() << "setPose";
  qInfo() << " pose" << pose;
  qInfo() << " volume" << volume;
  qInfo() << " frame" << frame;
  this->Pose = pose;
}

// --------------------------------------------------------------------------
void AutoscoperMockMainWindow::setBackground(double threshold)
{
  qInfo() << "setBackground";
  qInfo() << " threshold" << threshold;
}

// --------------------------------------------------------------------------
std::vector <double> AutoscoperMockMainWindow::getNCC(unsigned int volumeID, double* xyzpr)
{
  qInfo() << "getNCC";
  qInfo() << " volumeID" << volumeID;
  qInfo() << " xyzpr" << xyzpr;
  std::vector<double> correlations;
  correlations.push_back(0.5);
  return correlations;
}

// --------------------------------------------------------------------------
void AutoscoperMockMainWindow::saveFullDRR()
{
  qInfo() << "saveFullDRR";
}

// --------------------------------------------------------------------------
std::vector <unsigned char> AutoscoperMockMainWindow::getImageData(unsigned int volumeID, unsigned int camera, double* xyzpr, unsigned int &width, unsigned int &height)
{
  qInfo() << "getImageData";
  qInfo() << " volumeID" << volumeID;
  qInfo() << " camera" << camera;
  qInfo() << " xyzpr" << xyzpr;
  qInfo() << " width" << width;
  qInfo() << " height" << height;

  // Generate image with four quadrants
  std::vector<unsigned char> img_Data;
  for (unsigned int w_idx = 0; w_idx < width; ++w_idx)
    {
    for (unsigned int h_idx = 0; h_idx < height; ++h_idx)
      {
      bool width_index_right = w_idx > width / 2;
      bool height_index_bottom =  h_idx > height / 2;
      bool toggle = false;
      if ((width_index_right and height_index_bottom) || (not width_index_right and not height_index_bottom))
        {
        toggle = true;
        }
      img_Data.push_back((unsigned char)(255 * toggle));
      }
    }

  return img_Data;
}

// --------------------------------------------------------------------------
void AutoscoperMockMainWindow::optimizeFrame(int volumeID, int frame, int dframe, int repeats, int opt_method, unsigned int max_iter, double min_limit, double max_limit, int cf_model, unsigned int stall_iter)
{
  qInfo() << "optimizeFrame";
  qInfo() << " volumeID" << volumeID;
  qInfo() << " frame" << frame;
  qInfo() << " dframe" << dframe;
  qInfo() << " repeats" << repeats;
  qInfo() << " opt_method" << opt_method;
  qInfo() << " max_iter" << max_iter;
  qInfo() << " min_limit" << min_limit;
  qInfo() << " max_limit" << max_limit;
  qInfo() << " cf_model" << cf_model;
  qInfo() << " stall_iter" << stall_iter;
}
