### CONNECTION 
import socket
from struct import pack

client  = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('socket instantiated')
HOST = 'localhost'
PORT = 30007
client.connect((HOST, PORT))
print('socket connected')

conn_type = 1
pp = b"C:\Dev\autoscoper-git\build\install\bin\Release\sample_data\wrist.cfg"
data = pack('i 66s', conn_type, pp)
client.sendall(data)