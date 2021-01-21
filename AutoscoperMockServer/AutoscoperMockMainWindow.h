
// Based on autoscoper/src/ui/AutoscoperMainWindow.h

// Qt includes
#include <QObject>
#include <QString>

// STD includes
#include <vector>

class AutoscoperMockMainWindow : public QObject
{
  Q_OBJECT
public:
  AutoscoperMockMainWindow();

  void setFrame(int frame);

  //For socket
  void openTrial(QString filename);
  void load_tracking_results(QString filename, bool save_as_matrix, bool save_as_rows, bool save_with_commas, bool convert_to_cm, bool convert_to_rad, bool interpolate, int volume = -1);
  void save_tracking_results(QString filename, bool save_as_matrix, bool save_as_rows, bool save_with_commas, bool convert_to_cm, bool convert_to_rad, bool interpolate, int volume = -1);
  void loadFilterSettings(int camera, QString filename);
  std::vector<double> getPose(unsigned int volume, unsigned int frame);
  void setPose(std::vector<double> pose, unsigned int volume, unsigned int frame);
  void setBackground(double threshold);
  std::vector <double> getNCC(unsigned int volumeID, double* xyzpr);
  void saveFullDRR();
  std::vector <unsigned char> getImageData(unsigned int volumeID, unsigned int camera, double* xyzpr, unsigned int &width, unsigned int &height);
  void optimizeFrame(int volumeID, int frame, int dframe, int repeats, int opt_method, unsigned int max_iter, double min_limit, double max_limit, int cf_model, unsigned int stall_iter);

private:
  // For getPose/setPose
  int Frame { -1 };
  std::vector<double> Pose;

private:
  Q_DISABLE_COPY(AutoscoperMockMainWindow)
};

class AutoscoperMainWindow : public AutoscoperMockMainWindow
{
};
