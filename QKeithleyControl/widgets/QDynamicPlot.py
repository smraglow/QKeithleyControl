# ---------------------------------------------------------------------------------
# 	QDynamicPlot.py
# 	Copyright (C) 2019 mwchalmers
#	mwchalmers@protonmail.com
# ---------------------------------------------------------------------------------
# 
# 	Permission is hereby granted, free of charge, to any person obtaining a copy
# 	of this software and associated documentation files (the "Software"), to deal
# 	in the Software without restriction, including without limitation the rights
# 	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# 	copies of the Software, and to permit persons to whom the Software is
# 	furnished to do so, subject to the following conditions:
# 	
# 	The above copyright notice and this permission notice shall be included in all
# 	copies or substantial portions of the Software.
# 	
# 	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# 	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# 	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# 	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# 	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# 	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# 	SOFTWARE.
#

#!/usr/bin/env python
import numpy as np

# Import QT backends
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QCheckBox, QLabel, QMessageBox,  QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

# Import matplotlibQT backends
import os 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt


class QDynamicPlot(QWidget):

	# Note that from the main app, we must pass the pointer of the app 
	# instance to dynamic plotdynamic_plot.dynamic_plot(self)
	def __init__(self, *args, **kwargs):

		super(QDynamicPlot, self).__init__(*args, **kwargs)

		# Variables for axes labels
		self.xlabel = None
		self.ylabel = None
		self.tlabel = None

		# Flag for twinx dynamic plots
		self.twinx = False	

		# List of handles of plot into
		self.hlist = []

		self.figure  = plt.figure(figsize=(8,5))
		self.canvas  = FigureCanvas(self.figure)
		self.toolbar = NavigationToolbar(self.canvas, self)
		self.refresh = QPushButton("Clear Data")
		self.refresh.clicked.connect(self.refresh_axes)

		self.hlayout1 = QHBoxLayout()		
		self.hlayout1.addWidget(self.toolbar)
		self.hlayout1.addWidget(self.refresh)

		self.hlayout2 = QHBoxLayout()
		self.hlayout2.addWidget(self.canvas)
		
		self.layout = QVBoxLayout()
		self.layout.addLayout(self.hlayout2)
		self.layout.addLayout(self.hlayout1)

		self.setLayout( self.layout )

		# External handle for dialog answer
		self.msg_clear = None

		# Create Icon for QMessageBox
		self.icon = QIcon(os.path.join(os.path.dirname(os.path.realpath(__file__)), "python.ico"))	

	# Expose refresh axes
	def refresh_axes(self):
		
		# Only ask to redraw if there is data present
		if self.hlist != []:

			msg = QMessageBox()
			msg.setIcon(QMessageBox.Information)
			msg.setText("Clear all measurement data?")
			msg.setWindowTitle("QDynamicPlot")
			msg.setWindowIcon(self.icon)
			msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
			self.msg_clear = msg.exec_()

			if self.msg_clear == QMessageBox.Yes:
				self._refresh_axes()		
	
		else:
			self._refresh_axes()		

	# Internal method to clear axes		
	def _refresh_axes(self):

		self.figure.clear()
		self.hlist=[]

		# Call add axes with self._twinx
		self.add_axes(self.twinx)

	# Add axes object to widget and draw figure
	def add_axes(self, _twinx = False):	

		# Set twinx flag
		self.twinx = _twinx

		# Add axes and set axes labels
		self.ax = self.figure.add_subplot(111)

		if self.xlabel is not None:
			self.ax.set_xlabel(self.xlabel)

		if self.ylabel is not None:
			self.ax.set_ylabel(self.ylabel)

		# Generate twinned axes
		if self.twinx == True:

			# Create twinned axes
			self.ax_twin = self.ax.twinx()	

			if self.tlabel is not None:
				self.ax_twin.set_ylabel(self.tlabel)
	
		self._draw_canvas()

	# Add axes labels
	def set_axes_labels(self, _xlabel, _ylabel, _tlabel=None):
		self.xlabel = str(_xlabel)
		self.ylabel = str(_ylabel)
		if _tlabel is not None:
			self.tlabel = str(_tlabel)
	
	# Add trace
	def add_handle(self, _axes_index=0, _color=None):

		if _color is not None:
			h, = self.figure.axes[_axes_index].plot([], [], color=_color)
		else:
			h, = self.figure.axes[_axes_index].plot([], [])

		self.hlist.append(h)
		return h	

	# Update trace	
	def update_handle(self, h, x_value, y_value, _axes_index=0):

		h.set_xdata(np.append(h.get_xdata(), x_value))
		h.set_ydata(np.append(h.get_ydata(), y_value))
		
	def _draw_canvas(self):

		for _ in range(len(self.figure.axes)):
			self.figure.axes[_].relim()
			self.figure.axes[_].autoscale_view()
			self.figure.axes[_].ticklabel_format(style='sci', scilimits=(0,0), axis='y', useOffset=False)

		if self.twinx:
			plt.subplots_adjust(left=0.15, right=0.85, top=0.90, bottom=0.10)
		else: 
			plt.subplots_adjust(left=0.15, right=0.90, top=0.90, bottom=0.10)

		self.figure.canvas.draw()
		self.figure.canvas.flush_events()
