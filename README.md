📦 Location-Based Object Validator using YOLO and Streamlit
A simple prototype for context-aware object detection in manufacturing environments.
This app uses a YOLOv5 model to detect objects in uploaded images and verifies whether they are allowed in a specific workspace, such as a room, conveyor, or rack area.

🔍 Use Case
In modern manufacturing, misplaced tools or parts — especially on conveyors, in clean rooms, or on the wrong racks — can lead to:

⚙️ Equipment damage

🧪 Product contamination

🧱 Assembly errors

📉 Regulatory violations

⏱️ Lost time and rework

This app explores how lightweight AI + visual rules can help identify violations early and generate audit reports for traceability and compliance.

🧠 Features
🗂️ Location selection (Room A, Conveyor 1, etc.)

🎛️ Custom rules for allowed object classes per location

🧠 Object detection using YOLOv5

🚨 Highlights violations (e.g., tool not allowed in clean area)

📄 PDF report export with detected objects and policy status

🖥️ Streamlit interface — fast, interactive, and deployable

🚀 Getting Started
1. 🧬 Clone the repo
bash
  git clone https://github.com/Ivan2911/object-validator-app.git
  cd object-validator-app

2. 📦 Install dependencies
bash
  pip install -r requirements.txt

4. 🧠 Add YOLOv5 Model Weights
You need to add a [YOLOv5 model](https://github.com/ultralytics/yolov5) file (e.g., yolov5s.pt) to the project so the app can perform object detection.

    🔽  Download the weights manually
    Go to the Ultralytics YOLOv5 release page
    Download the file named yolov5s.pt
    Create a **models** folder in your project root (if it doesn't exist)
    Move the downloaded file into that folder

4. 🏃 Run the app
bash
streamlit run app.py


📄 Example
🏷️ Select a room (e.g., Conveyor 2)

✅ Choose allowed object types (e.g., box, person, tape)

📤 Upload a photo

🤖 App detects objects and flags any violations

🧾 Download a PDF report for documentation

🧪 Status
This is a research prototype, not a production system.
It’s intended for experimentation and academic/industrial collaboration in:

🕵️‍♂️ Visual inspection

👷‍♀️ Human-in-the-loop quality control

🧠 Smart factory safety systems

📚 Future Ideas
🧪 Model fine-tuning with factory-specific data

📸 Webcam/IP camera support

🚨 Real-time alerts or alarms

📦 Part/rack location validation via QR or RFID

🖥️ Edge deployment with Jetson or Raspberry Pi

🤝 Contributing
Pull requests are welcome!
If you have ideas, feedback, or real-world use cases, feel free to open an issue or get in touch.

🧑‍🔬 Author
  Ivandros Awom
🔗 Connect on [LinkedIn](https://www.linkedin.com/in/ivandrosawom) 

