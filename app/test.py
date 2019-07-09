# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 06:37:00 2019

@author: viswanathan.a
"""

from flask import Flask
app = Flask(__name__)

@app.route('/')
def dhineTrend():
   return "hello python"

if __name__ == '__main__':
   app.run()