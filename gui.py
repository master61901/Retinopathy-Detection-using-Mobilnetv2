# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(813, 503)
        self.imgLabel = QtWidgets.QLabel(Dialog)
        self.imgLabel.setGeometry(QtCore.QRect(10, 10, 640, 480))
        self.imgLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.imgLabel.setText("")
        self.imgLabel.setObjectName("imgLabel")
        self.groupBox = QtWidgets.QGroupBox(Dialog)
        self.groupBox.setGeometry(QtCore.QRect(660, 140, 141, 351))
        self.groupBox.setObjectName("groupBox")
        self.BrowseButton = QtWidgets.QPushButton(self.groupBox)
        self.BrowseButton.setGeometry(QtCore.QRect(10, 110, 121, 61))
        self.BrowseButton.setObjectName("BrowseButton")
        self.TrainButton = QtWidgets.QPushButton(self.groupBox)
        self.TrainButton.setGeometry(QtCore.QRect(10, 30, 121, 51))
        self.TrainButton.setObjectName("TrainButton")
        self.ExitButton = QtWidgets.QPushButton(self.groupBox)
        self.ExitButton.setGeometry(QtCore.QRect(10, 290, 121, 51))
        self.ExitButton.setObjectName("ExitButton")
        self.DetectRecogniseButton = QtWidgets.QPushButton(self.groupBox)
        self.DetectRecogniseButton.setGeometry(QtCore.QRect(10, 200, 121, 51))
        self.DetectRecogniseButton.setObjectName("DetectRecogniseButton")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(660, 10, 141, 121))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setText("")
        self.label.setObjectName("label")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Diabetic Retinopathy Detection using machine Learning"))
        self.groupBox.setTitle(_translate("Dialog", "Process"))
        self.BrowseButton.setText(_translate("Dialog", "Browse Test Image"))
        self.TrainButton.setText(_translate("Dialog", "Training"))
        self.ExitButton.setText(_translate("Dialog", "Exit"))
        self.DetectRecogniseButton.setText(_translate("Dialog", " Recognise"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

