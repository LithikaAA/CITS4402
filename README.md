# CITS4402 Computer Vision Project: Face Detection and Matching

## Overview
This project implements an automated system for face detection, alignment, and identity matching. It is designed to process surveillance footage by detecting faces across multiple camera feeds and clustering them by unique identities. 

## Group Members
* **Jason Rarey** - 22681384
* **Lithika Senthil Kumar** - 23812347

## Features
1.  **Graphical User Interface (GUI):** Built with Tkinter to allow for single-image and bulk-folder processing.
2.  **Face Detection:** Detects up to 4 faces per image, utilizing skin color segmentation to reduce false positives.
3.  **Facial Landmark Detection:** Identifies the left eye-center, right eye-center, and nose tip.
4.  **Facial Alignment:** Applies similarity transformations to align, crop, and resize faces to 125x125 pixels.
5.  **Identity Clustering:** Bulk processes images, extracts facial features, and utilizes density-based clustering (e.g., DBSCAN) to find the number of unique identities without predefined cluster counts.

## Setup and Installation

**1. Clone the repository:**
```bash
git clone [https://github.com/LithikaAA/CITS4402.git](https://github.com/LithikaAA/CITS4402.git)
cd CITS4402