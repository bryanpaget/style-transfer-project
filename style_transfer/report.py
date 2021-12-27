#!/usr/bin/env python

import platform
import datetime

from lshw.client import ListHardware

"""
TODO: Final Report
Each artwork comes with a report.

The report tells you when it was made (timestamp), the model, the hardware it was run on, the software made to produce
it, and other interesting things. All put into a PDF. Like a certificate that tells you all about how it was made.

Title
Date
Hardware
Software
OS
Location

Style Image
Content Image

Parameters
Code

Resulting Image

Artist's Cryptographic signature (public key)

"""


class Certificate:
    def __init__(self):
        self.title = ""
        self.data = datetime.datetime.now()
        self.hardware = self.list_hardware()

    @staticmethod
    def list_hardware(self):
        list_hardware = ListHardware()
        return list_hardware.read_data(list_hardware.get_nic_data())
