# _*_ coding: utf-8 _*_
# @File        : easyserial.py
# @Time        : 2021/10/17 13:31
# @Author      : SaleJuice
# @E-Mail      : linxzh@shanghaitech.edu.cn
# @Institution : LIMA Lab, ShanghaiTech University, China
# @SoftWare    : PyCharm

import sys
import time
import serial
import serial.tools.list_ports


class LinuxBackground:

    def __init__(self):
        assert ('linux' in sys.platform), "Linux Only!"
        self.port_list = []
        self.ser = None

    def search(self, portx=""):
        port_list = list(serial.tools.list_ports.comports())
        for i in range(len(port_list)):
            self.port_list.append(port_list[i].name)

        search_flag = False
        if portx == "":
            search_flag = True
        else:
            for i in range(len(self.port_list)):
                if portx == self.port_list[i]:
                    search_flag = True
                    break

        return self.port_list, search_flag


    def connect(self, portx, baud=115200, timeout=3):
        connect_flag = False
        if self.search(portx)[1]:
            try:
                self.ser = serial.Serial("/dev/"+portx, baud, timeout=timeout)
                connect_flag = True
            except Exception as e:
                print("Use command: 'sudo chmod 666 /dev/ttyUSB1' in terminal and try again.")
                connect_flag = False
        else:
            connect_flag = False

        return connect_flag

    def close(self):
        close_flag = False
        if self.ser == None:
            close_flag = False
        else:
            self.ser.close()
            close_flag = True

        return close_flag

    def read(self, type="all", timeout=None):
        read_flag = False
        statime = time.perf_counter()
        endtime = 0
        if self.ser == None:
            count = 0
            read_result = None
            read_flag = False
        else:
            while True:
                if self.ser.in_waiting:
                    count = self.ser.in_waiting
                    if type == "all":
                        read_result = self.ser.read(self.ser.in_waiting).decode("ASCII", "ignore")
                    if type == "line":
                        read_result = self.ser.readline().decode("ASCII", "ignore")
                    if type == "lines":
                        read_result = self.ser.readlines().decode("ASCII", "ignore")
                    read_flag = True
                    break
                endtime = time.perf_counter()
                if timeout != None and endtime - statime > timeout:
                    count = 0
                    read_result = None
                    read_flag = False
                    break

        return count, read_result, read_flag

    def write(self, data):
        write_flag = False
        count = self.ser.write(data.encode("ASCII"))
        if count == 0:
            write_flag = False
        else:
            write_flag = True

        return count, write_flag


class WindowsBackground:

    def __init__(self):
        assert ('win' in sys.platform), "Windows Only!"
        self.port_list = []
        self.ser = None

    def search(self, portx=""):
        port_list = list(serial.tools.list_ports.comports())
        for i in range(len(port_list)):
            self.port_list.append(port_list[i].name)

        search_flag = False
        if portx == "":
            search_flag = True
        else:
            for i in range(len(self.port_list)):
                if portx == self.port_list[i]:
                    search_flag = True
                    break

        return self.port_list, search_flag


    def connect(self, portx, baud=115200, timeout=3):
        connect_flag = False
        if self.search(portx)[1]:
            try:
                self.ser = serial.Serial(portx, baud, timeout=timeout)
                connect_flag = True
            except Exception as e:
                connect_flag = False
        else:
            connect_flag = False

        return connect_flag

    def close(self):
        close_flag = False
        if self.ser == None:
            close_flag = False
        else:
            self.ser.close()
            close_flag = True

        return close_flag

    def read(self, type="all", timeout=None):
        read_flag = False
        statime = time.perf_counter()
        endtime = 0
        if self.ser == None:
            count = 0
            read_result = None
            read_flag = False
        else:
            while True:
                if self.ser.in_waiting:
                    count = self.ser.in_waiting
                    if type == "all":
                        read_result = self.ser.read(self.ser.in_waiting).decode("ASCII", "ignore")
                    if type == "line":
                        read_result = self.ser.readline().decode("ASCII", "ignore")
                    if type == "lines":
                        read_result = self.ser.readlines().decode("ASCII", "ignore")
                    read_flag = True
                    break
                endtime = time.perf_counter()
                if timeout != None and endtime - statime > timeout:
                    count = 0
                    read_result = None
                    read_flag = False
                    break

        return count, read_result, read_flag

    def write(self, data):
        write_flag = False
        count = self.ser.write(data.encode("ASCII"))
        if count == 0:
            write_flag = False
        else:
            write_flag = True

        return count, write_flag



if __name__ == '__main__':
    ser = WindowsBackground()
    print(ser.connect("COM4"))
    print(ser.read("all"))
    print(ser.write("(888,0,0)\n"))
    print(ser.close())
