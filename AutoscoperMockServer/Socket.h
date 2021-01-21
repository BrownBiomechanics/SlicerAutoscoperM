
// Based on autoscoper/src/net/Socket.h

#pragma once

#ifndef SOCKET_H
#define SOCKET_H

#include <QTcpServer>
#include <QObject>

class AutoscoperMainWindow;

class Socket : public QObject
{
  Q_OBJECT

public:
  Socket(AutoscoperMainWindow* mainwindow, unsigned long long int listenPort);
  ~Socket();

private:
  QTcpServer *tcpServer;
  std::vector<QTcpSocket *> clientConnections;
  void handleMessage(QTcpSocket * connection, char* data, qint64 length);
private:
  AutoscoperMainWindow* m_mainwindow;

private slots:
  void createNewConnection();
  void deleteConnection();
  void reading();
};

#endif // SOCKET_H

