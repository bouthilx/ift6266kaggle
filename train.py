#! /usr/bin/env python

from pylearn2.utils import serial
import sys

if __name__ == "__main__":

    train_obj = serial.load_train_file(sys.argv[1])
    train_obj.main_loop()
