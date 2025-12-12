# Daleel: Person Re-Identification System for Finding and Tracking Lost People
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
* **Person Re-Identification (Re-ID):** Tracks an individual with a consistent ID as they move across different cameras.
* **Person Search**: searches for a person using pictures and/or a textual description.
* **Intelligent Detection:**
    * **Emotion Detection:** Marks individuals who appear lost or disoriented.
    * **Vulnerable Persons:** Identifies and flags elderly and disabled individuals.
* **Crowd Counting:** Provides a semi-real-time count of people.
  
### System Pipeline

When the system starts, it follows the pipline below:

<img width="955" height="737" alt="image" src="https://github.com/user-attachments/assets/4a4b77e7-2800-4d8c-bfc1-d1cd2c8cec06" />

### Person Search

To look for a certain individual using some pictures and/or textual description:

<img width="1271" height="490" alt="image" src="https://github.com/user-attachments/assets/1b2fe4b3-ae0c-4cf8-a518-d4493fdeae4b" />


## Running The System

Our system runs in python `3.10` and `3.11`.

After cloning this repo, install the needed packages

```bash
cd Daleel
pip install -r requirements.txt
```

To start the system, simply run:

```bash
python -m streamlit run Streamlit\app.py
```

You should see the User Interface (UI):

<img width="1919" height="699" alt="image" src="https://github.com/user-attachments/assets/cb13993d-ca17-40be-aee3-0618a7b23b5f" />




