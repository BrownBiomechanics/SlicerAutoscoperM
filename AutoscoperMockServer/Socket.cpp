
// Based on autoscoper/src/net/Socket.cpp

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <iostream>
#include <QTcpSocket>

#include "AutoscoperMockMainWindow.h"
#include "Socket.h"

Socket::Socket(AutoscoperMainWindow* mainwindow, unsigned long long int listenPort) : m_mainwindow(mainwindow)
{
  tcpServer = new QTcpServer();

  connect(tcpServer, &QTcpServer::newConnection, this, &Socket::createNewConnection);
  tcpServer->listen(QHostAddress::LocalHost, listenPort);
} 

Socket::~Socket()
{
  for (auto &a : clientConnections){
    a->disconnectFromHost();
  }
}

void Socket::handleMessage(QTcpSocket * connection, char* data, qint64 length)
{
  unsigned char message_type = data[0];

  switch (message_type)
  {
  case 1:    
    {
      //load trial
      std::string filename = std::string(&data[1],length-1);

      std::cerr << "load trial " << filename.c_str() << std::endl;
      m_mainwindow->openTrial(QString(filename.c_str()));

      connection->write(QByteArray(1, 1));
    }
    break;
  case 2:
    //load tracking data
    {
      qint32* volume = reinterpret_cast<qint32*>(&data[1]);
      std::string filename = std::string(&data[5], length - 5);
      std::cerr << "load tracking data Volume " << *volume << " : " << filename.c_str() << std::endl;

      m_mainwindow->load_tracking_results(QString(filename.c_str()), true, true, true, false, false, false, *volume);

      connection->write(QByteArray(1, 2));
    }
    break;
  case 3:
    //save tracking data
    {
      qint32* volume = reinterpret_cast<qint32*>(&data[1]);
      std::string filename = std::string(&data[5], length - 5);

      std::cerr << "save tracking data Volume " << *volume << " : " << filename.c_str() << std::endl;
      m_mainwindow->save_tracking_results(QString(filename.c_str()), true, true, true, false, false, false, *volume);

      connection->write(QByteArray(1, 3));
    }
    break;
  case 4:
    //load filter settings
    {
      qint32* camera = reinterpret_cast<qint32*>(&data[1]);
      std::string filename = std::string(&data[5], length - 5);

      std::cerr << "load filter settings for camera " << *camera << " : " << filename.c_str() << std::endl;
      m_mainwindow->loadFilterSettings(*camera, QString(filename.c_str()));

      connection->write(QByteArray(1, 4));
    }
    break;
  case 5:
    //set current frame
    {
      qint32* frame = reinterpret_cast<qint32*>(&data[1]);

      std::cerr << "set frame to " << *frame << std::endl;
      m_mainwindow->setFrame(*frame);

      connection->write(QByteArray(1, 5));
    }
    break;
  case 6:
    //get Pose
    {
      qint32* volume = reinterpret_cast<qint32*>(&data[1]);
      qint32* frame = reinterpret_cast<qint32*>(&data[5]);

      std::cerr << "get pose for volume " << *volume << " frame " << *frame << std::endl;
      std::vector<double> pose = m_mainwindow->getPose(*volume,*frame);

      char * ptr = reinterpret_cast<char*>(&pose[0]);
      QByteArray array = QByteArray(1, 6);
      array.append(ptr, sizeof(double) * 6);
      connection->write(array);
    }
    break;
  case 7:
    //set Pose
    {
      qint32* volume = reinterpret_cast<qint32*>(&data[1]);
      qint32* frame = reinterpret_cast<qint32*>(&data[5]);
      double * pose_data = reinterpret_cast<double*>(&data[9]);
      std::vector<double> pose;
      pose.assign(pose_data, pose_data + 6);

      std::cerr << "set pose for volume " << *volume << " frame " << *frame;
      for (auto a : pose)
        std::cerr << " " << a;
      std::cerr << std::endl;
      m_mainwindow->setPose(pose, *volume, *frame);

      connection->write(QByteArray(1, 7));
    }
    break;
  case 8:
    //get NCC
    {
      qint32* volume = reinterpret_cast<qint32*>(&data[1]);
      double * pose_data = reinterpret_cast<double*>(&data[5]);
      std::vector<double> pose;
      pose.assign(pose_data, pose_data + 6);

      std::vector<double> ncc = m_mainwindow->getNCC(*volume, &pose_data[0]);

      //Return double
      char * ptr = reinterpret_cast<char*>(&ncc[0]);
      QByteArray array = QByteArray(1, 8);
      array.append(QByteArray(1, ncc.size()));
      array.append(ptr, sizeof(double) * ncc.size());
      connection->write(array);
    }
  break;
  case 9:
    //set Background
    {
      double * threshold = reinterpret_cast<double*>(&data[1]);

      std::cerr << "set background " << *threshold << std::endl;
      m_mainwindow->setBackground(*threshold);

      connection->write(QByteArray(1, 9));
    }
  break;
  case 10:
    //get the image - cropped
    {
      qint32* volume = reinterpret_cast<qint32*>(&data[1]);
      qint32* camera = reinterpret_cast<qint32*>(&data[5]);
      double * pose_data = reinterpret_cast<double*>(&data[9]);
      
      std::cerr << "Read images for volume " << *volume << " and camera " << *camera << std::endl;

      std::vector<double> pose;
      pose.assign(pose_data, pose_data + 6);
      unsigned int width, height;
      std::vector<unsigned char> img_Data = m_mainwindow->getImageData(*volume,*camera, &pose_data[0], width, height );

      QByteArray array = QByteArray(1, 10);
      char * ptr = reinterpret_cast<char*>(&width);
      array.append(ptr, sizeof(qint32));
        ptr = reinterpret_cast<char*>(&height);
      array.append(ptr, sizeof(qint32));
      ptr = reinterpret_cast<char*>(&img_Data[0]);
      array.append(ptr, img_Data.size());
      connection->write(array);
      std::cerr << width << " " << height << " " << img_Data.size() << std::endl;
    
    }
    break;
  case 11:
    //optimize from matlab
    {
      qint32* volumeID = reinterpret_cast<qint32*>(&data[1]);
      qint32* frame = reinterpret_cast<qint32*>(&data[5]);
      qint32* repeats = reinterpret_cast<qint32*>(&data[9]);
      qint32* max_iter = reinterpret_cast<qint32*>(&data[13]);
      double* min_limit = reinterpret_cast<double*>(&data[17]);
      double* max_limit = reinterpret_cast<double*>(&data[25]);
      qint32* stall_iter = reinterpret_cast<qint32*>(&data[33]);

      qint32 dframe = 1;// reinterpret_cast<qint32*>(&data[5]);
      qint32 opt_method = 0;//reinterpret_cast<qint32*>(&data[25]);
      qint32 cf_model = 0;//reinterpret_cast<qint32*>(&data[41]);

      std::cerr << "Running optimization from autoscoper for frame #" << *frame << std::endl;

      m_mainwindow->optimizeFrame(*volumeID, *frame, dframe, *repeats,
        opt_method,
        *max_iter, *min_limit, *max_limit,
        cf_model, *stall_iter);

      connection->write(QByteArray(1, 11));
    }
    break;

  case 12:
    //save full drr image
    {
      std::cerr << "Saving the full DRR image: " << std::endl;

      m_mainwindow->saveFullDRR();

      connection->write(QByteArray(1, 12));
    }
    break;

  default:
    std::cerr << "Cannot handle message" << std::endl;
    connection->write(QByteArray(1,0));
    break;
  }
}

void Socket::createNewConnection()
{
  std::cerr << "New Matlab Client is Connected..." << std::endl;
  QTcpSocket *clientConnection = tcpServer->nextPendingConnection();
  connect(clientConnection, &QAbstractSocket::disconnected, this, &Socket::deleteConnection);
  connect(clientConnection, &QIODevice::readyRead, this, &Socket::reading);

  clientConnections.push_back(clientConnection);
}

void Socket::deleteConnection()
{
  //std::cerr << "client disconnected" << std::endl;
  QTcpSocket * obj = dynamic_cast<QTcpSocket *>(sender());
  if (obj)
  {
    clientConnections.erase(std::remove(clientConnections.begin(), clientConnections.end(), obj), clientConnections.end());
  }
  obj->deleteLater();
}

void Socket::reading()
{
  QTcpSocket * obj = dynamic_cast<QTcpSocket *>(sender());
  if (obj)
  {
    qint64 avail = obj->bytesAvailable();
    char *data = new char[avail];
    obj->read(data, avail);
    handleMessage(obj, data, avail);
    delete[] data;
  }
}

