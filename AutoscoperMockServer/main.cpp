
#include <QCoreApplication>
#include <QDebug>

#include "AutoscoperMockMainWindow.h"
#include "Socket.h"

int main(int argc, char* argv[])
{
  QCoreApplication app(argc, argv);

  int port = 30007;

  qInfo() << "AutoscoperMockServer listening on port" << port;

  AutoscoperMainWindow widget;
  Socket* socket = new Socket(&widget, port);

  return app.exec();
}
