# 🕌 AI-Powered Support System for Security

An intelligent system to assist security authorities at the Prophet's Mosque in identifying, locating, and managing cases of lost or disoriented individuals in real-time using AI and computer vision.

## 🎯 The Problem

In large, densely crowded environments like Al-Masjid An-Nabawi, manually monitoring surveillance feeds to find a lost or disoriented person is extremely challenging. This traditional method often leads to:

* **Delayed Identification:** It takes a long time to spot and confirm a missing person.
* **High Workload:** Security staff face a heavy, continuous burden of monitoring hundreds of cameras.
* **Limited Accuracy:** Tracking an individual as they move between different camera views is difficult and prone to error.
* **Inefficient Resource Use:** A significant number of personnel are required for manual monitoring.

## 💡 Our Solution

This project is an **AI-Powered Support System for Security** that addresses these challenges. Unlike manual monitoring, our system intelligently detects, locates, and notifies security personnel about lost individuals.

### Core System & AI Capabilities

* **Live Stream Analysis:** Ingests and analyzes live video feeds from surveillance cameras.
* **Person Re-Identification (Re-ID):** Tracks an individual with a consistent ID as they move across different cameras (within a 15-minute window).
* **Intelligent Detection:**
    * **Emotion Detection:** Marks individuals who appear lost or disoriented.
    * **Vulnerable Persons:** Identifies and flags elderly and disabled individuals.
    * **Object Detection:** Detects and flags prohibited objects.
* **Crowd Counting:** Provides a semi-real-time count of people within specific, defined areas.
* **Data Management:** Stores tracking data in 30-second clips and automatically purges data after 15 minutes of no appearance to maintain privacy and performance.

## 💻 Technology Stack

* **Backend Framework:** **Django**
* **Language:** Python
* **AI / Computer Vision:**
    * Person Re-Identification Models
    * Emotion Detection
    * Object Detection (e.g., YOLO)
    * OpenCV for video processing
* **Database:** (To be determined, e.g., PostgreSQL, SQLite)
* **Frontend:** (To be determined, e.g., HTML, CSS, JS)
