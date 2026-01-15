# Flower-detection-and-Separation-in-conveyor-belt

🌸 Flower Detection and Separation System on Conveyor Belt
📌 Project Overview

The Flower Detection and Separation System is an automated solution designed to identify and sort flowers in real time using computer vision and IoT. The system uses an ESP32-CAM mounted above a conveyor belt to capture images of flowers, while a Python-based image processing model detects flower types and controls the separation mechanism.

An ESP8266 is used to control the conveyor belt and actuators, enabling accurate and fast sorting with minimal human intervention.

🎯 Key Features

🌼 Real-time flower detection using ESP32-CAM

🧠 Image processing and classification using Python

🔁 Automated sorting on a conveyor belt

🌐 IoT-based control using ESP8266

⚙️ Precise separation using motors / actuators

⏱️ High-speed and accurate sorting

🤖 Reduces manual labor in flower processing

🛠️ Components Used

ESP8266 (NodeMCU)

ESP32-CAM

Conveyor Belt Mechanism

DC Motor / Servo Motor

Motor Driver Module

Power Supply

PC / Laptop (for Python processing)

Connecting Wires & Mechanical Frame

⚙️ Working Principle

Flowers are placed on the moving conveyor belt.

The ESP32-CAM captures real-time images of the flowers.

Images are sent to a Python application for processing.

The Python program detects and classifies flowers using image processing techniques.

Based on the detected flower type:

A control signal is sent to the ESP8266

The ESP8266 activates the appropriate actuator to separate the flower into the correct section.

The conveyor continues for continuous sorting.

🔁 System Flow
Flower on Conveyor
        ↓
ESP32-CAM → Image Capture
        ↓
Python Processing (Detection)
        ↓
Decision Signal
        ↓
ESP8266 → Motor / Actuator
        ↓
Flower Separation

🌍 Applications

Flower grading and sorting industries

Smart agriculture systems

Automated packaging units

Industrial conveyor sorting

IoT + computer vision research projects

🚀 Advantages

Accurate flower identification

Reduces manual sorting errors

Scalable for industrial use

Cost-effective automation

Real-time processing

📌 Future Enhancements

Deep learning-based flower classification

Multi-color and multi-species detection

Cloud monitoring dashboard

Robotic arm-based separation

Speed optimization for high-throughput sorting

📂 Repository Contents

ESP8266 firmware code

ESP32-CAM code

Python image processing script

Conveyor belt control logic

System diagrams and documentation
