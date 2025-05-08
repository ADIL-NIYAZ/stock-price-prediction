from flask import Flask, request, jsonify, render_template
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import os
import logging
from alpha_vantage.timeseries import TimeSeries